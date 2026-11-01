
"""
Build six-hour contiguous sequences for MIMIC BiLSTM and Transformer models.

Input:
    data/real/processed/splits/mimic_train.csv
    data/real/processed/splits/mimic_validation.csv
    data/real/processed/splits/mimic_test.csv

Output:
    data/real/processed/sequences/
        mimic_train_sequences.npz
        mimic_validation_sequences.npz
        mimic_test_sequences.npz
        sequence_generation_report.json

Each sequence contains six consecutive hourly observations from one ICU stay.
The target is the deterioration label at the final observation.
No interpolation or gap filling is performed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "real" / "processed" / "splits"
OUT_DIR = ROOT / "data" / "real" / "processed" / "sequences"

SPLITS = ("train", "validation", "test")
SEQ_LEN = 6
TARGET = "deterioration_label"
TIME_COL = "assessment_time"

EXCLUDED = {
    "subject_id",
    "hadm_id",
    "stay_id",
    TIME_COL,
    TARGET,
}


def make_sequences(df, feature_columns):

    X_parts = []
    y_parts = []
    subject_parts = []
    stay_parts = []
    time_parts = []

    candidate_windows = 0
    gap_breaks = 0

    # Process each ICU stay independently.
    for stay_id, stay in df.groupby("stay_id", sort=False):

        stay = stay.sort_values("_time").reset_index(drop=True)

        if len(stay) < SEQ_LEN:
            continue

        times = stay["_time"]

        # Identify exactly one-hour transitions.
        is_hourly = (
            times.diff()
            .dt.total_seconds()
            .div(3600)
            .eq(1)
        )

        # Start a new continuous run whenever a gap occurs.
        run_id = (~is_hourly).cumsum()

        for _, run in stay.groupby(run_id, sort=False):

            if len(run) < SEQ_LEN:
                continue

            values = run[feature_columns].to_numpy(
                dtype=np.float32,
                copy=True
            )

            labels = run[TARGET].to_numpy(
                dtype=np.int8,
                copy=True
            )

            subjects = run["subject_id"].to_numpy(
                dtype=np.int64,
                copy=False
            )

            stays = run["stay_id"].to_numpy(
                dtype=np.int64,
                copy=False
            )

            times_text = run[TIME_COL].astype(str).to_numpy()

            windows = len(run) - SEQ_LEN + 1
            candidate_windows += windows

            # Generate overlapping six-hour windows.
            seq = np.lib.stride_tricks.sliding_window_view(
                values,
                window_shape=SEQ_LEN,
                axis=0
            ).transpose(0, 2, 1).copy()

            X_parts.append(seq)

            # Target belongs to the final timestep.
            y_parts.append(labels[SEQ_LEN - 1:])

            subject_parts.append(subjects[SEQ_LEN - 1:])
            stay_parts.append(stays[SEQ_LEN - 1:])
            time_parts.append(times_text[SEQ_LEN - 1:])

        gap_breaks += int((~is_hourly).sum())

    if not X_parts:
        raise ValueError(
            "No valid six-hour sequences found."
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
        np.concatenate(stay_parts, axis=0),
        np.concatenate(time_parts, axis=0),
        candidate_windows,
        gap_breaks,
    )


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "sequence_length_hours": SEQ_LEN,
        "stride_hours": 1,
        "target": "deterioration_label at final sequence timestamp",
        "gap_policy": "Break sequence; no interpolation or forward filling",
        "split_unit": "subject_id",
        "missing_value_policy": (
            "Retain NaN; fit imputation using training sequences only"
        ),
        "splits": {}
    }

    reference_features = None

    for split in SPLITS:

        path = SPLIT_DIR / f"mimic_{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"\nLoading {split} split...")

        df = pd.read_csv(
            path,
            low_memory=False
        )

        required = {
            "subject_id",
            "stay_id",
            TIME_COL,
            TARGET
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        if df[TARGET].isna().any():
            raise ValueError(
                f"Missing labels in {split}"
            )

        # Select numeric predictors only.
        feature_columns = [
            col for col in df.columns
            if col not in EXCLUDED
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        non_numeric = [
            col for col in df.columns
            if col not in EXCLUDED
            and col not in feature_columns
        ]

        if non_numeric:
            raise ValueError(
                f"Non-numeric predictors: {non_numeric}"
            )

        if reference_features is None:
            reference_features = feature_columns

        elif feature_columns != reference_features:
            raise ValueError(
                "Feature columns differ between splits"
            )

        df["_time"] = pd.to_datetime(
            df[TIME_COL],
            errors="raise"
        )

        if df.duplicated(
            ["stay_id", TIME_COL]
        ).any():
            raise ValueError(
                f"Duplicate stay/time rows in {split}"
            )

        (
            X,
            y,
            subjects,
            stays,
            endpoint_times,
            windows,
            gap_breaks
        ) = make_sequences(
            df,
            feature_columns
        )

        output = OUT_DIR / f"mimic_{split}_sequences.npz"

        np.savez_compressed(
            output,
            X=X,
            y=y,
            subject_id=subjects,
            stay_id=stays,
            assessment_time=endpoint_times,
            feature_columns=np.asarray(
                feature_columns,
                dtype=str
            )
        )

        report["splits"][split] = {
            "source_rows": int(len(df)),
            "sequences": int(len(y)),
            "patients": int(np.unique(subjects).size),
            "stays": int(np.unique(stays).size),
            "positive_sequences": int(y.sum()),
            "negative_sequences": int((y == 0).sum()),
            "positive_rate": float(y.mean()),
            "features_per_timestep": int(len(feature_columns)),
            "sequence_shape": [
                int(X.shape[1]),
                int(X.shape[2])
            ],
            "candidate_windows": int(windows),
            "non_hourly_transitions": int(gap_breaks),
            "file": str(output)
        }

        print(
            f"{split}: "
            f"sequences={len(y):,}, "
            f"shape={X.shape}, "
            f"positive={int(y.sum()):,}, "
            f"negative={int((y == 0).sum()):,}"
        )

        print(
            f"Patients: {np.unique(subjects).size:,} | "
            f"Stays: {np.unique(stays).size:,}"
        )

        print(f"Saved: {output}")

    report["feature_columns"] = reference_features

    report_path = (
        OUT_DIR / "sequence_generation_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nSequence generation completed.")
    print("Report:", report_path)
    print("No model has been trained yet.")


if __name__ == "__main__":
    main()
