import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "real" / "processed"
INPUT = DATA / "mimic_features_with_vasopressors.csv"
OUT = DATA / "splits"

RANDOM_STATE = 42

def main():
    print("Loading feature dataset...")
    df = pd.read_csv(INPUT)

    required = {
        "subject_id", "stay_id",
        "assessment_time", "deterioration_label"
    }
    assert required.issubset(df.columns)
    assert df["deterioration_label"].notna().all()
    assert not df.duplicated(["stay_id", "assessment_time"]).any()

    # Stratify at patient level, not individual assessment level.
    patients = (
        df.groupby("subject_id")["deterioration_label"]
        .max()
        .reset_index(name="patient_positive")
    )

    train, temporary = train_test_split(
        patients,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=patients["patient_positive"]
    )

    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary["patient_positive"]
    )

    splits = {
        "train": set(train.subject_id),
        "validation": set(validation.subject_id),
        "test": set(test.subject_id)
    }

    # Verify that no patient occurs in multiple splits.
    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["validation"] & splits["test"])

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, patient_ids in splits.items():
        part = df[df.subject_id.isin(patient_ids)].copy()
        path = OUT / f"mimic_{name}.csv"
        part.to_csv(path, index=False)

        report[name] = {
            "rows": len(part),
            "patients": int(part.subject_id.nunique()),
            "stays": int(part.stay_id.nunique()),
            "positive": int((part.deterioration_label == 1).sum()),
            "negative": int((part.deterioration_label == 0).sum()),
            "positive_rate": float(part.deterioration_label.mean())
        }

        print(f"\n{name.upper()}")
        print(report[name])

    (OUT / "split_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPatient-level splitting completed.")
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()