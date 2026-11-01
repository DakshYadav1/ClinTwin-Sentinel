
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data/real/processed"

FILES = {
    "cardiovascular": "hourly_cardiovascular_sofa.csv",
    "respiratory": "hourly_respiratory_sofa.csv",
    "coagulation": "hourly_coagulation_sofa.csv",
    "liver": "hourly_liver_sofa.csv",
    "renal": "hourly_renal_sofa.csv",
    "neurological": "hourly_neurological_sofa.csv",
}

SCORE_COLUMNS = {
    "cardiovascular": "cardiovascular_sofa",
    "respiratory": "respiratory_sofa",
    "coagulation": "coagulation_sofa",
    "liver": "liver_sofa",
    "renal": "renal_sofa",
    "neurological": "neurological_sofa",
}

TOTAL_OUTPUT = DATA_DIR / "hourly_total_sofa.csv"
LABEL_OUTPUT = DATA_DIR / "mimic_deterioration_dataset.csv"
REPORT_OUTPUT = DATA_DIR / "total_sofa_label_report.json"

KEYS = ["stay_id", "assessment_time"]


# ============================================================
# 1. LOAD AND MERGE SIX COMPONENTS
# ============================================================

print("Loading cardiovascular SOFA as the base schedule...")

base_file = DATA_DIR / FILES["cardiovascular"]

df = pd.read_csv(
    base_file,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "assessment_time",
        "cardiovascular_sofa",
    ],
    parse_dates=["assessment_time"],
)

df = df.drop_duplicates(KEYS)

print(f"Base assessments: {len(df):,}")

for component, filename in FILES.items():

    if component == "cardiovascular":
        continue

    score_column = SCORE_COLUMNS[component]

    print(f"\nMerging {component} SOFA...")

    part = pd.read_csv(
        DATA_DIR / filename,
        usecols=KEYS + [score_column],
        parse_dates=["assessment_time"],
    )

    if part.duplicated(KEYS).any():
        raise ValueError(
            f"Duplicate assessment keys in {filename}"
        )

    df = df.merge(
        part,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    print(
        f"Rows after merging {component}: {len(df):,}"
    )


# ============================================================
# 2. VALIDATE COMPONENTS
# ============================================================

print("\nValidating SOFA components...")

components = list(SCORE_COLUMNS.values())

for column in components:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    invalid = df[column].notna() & ~df[column].between(0, 4)

    if invalid.any():
        raise ValueError(
            f"Invalid scores found in {column}: {invalid.sum()}"
        )

df = df.sort_values(
    ["stay_id", "assessment_time"],
    kind="mergesort",
).reset_index(drop=True)

df["available_component_count"] = (
    df[components].notna().sum(axis=1)
)

df["sofa_complete"] = (
    df["available_component_count"] == 6
)

# Do not convert missing components to zero.
df["total_sofa"] = df[components].sum(
    axis=1,
    min_count=6,
)

print("\nComponent coverage:")

for column in components:

    count = int(df[column].notna().sum())

    print(
        f"{column}: {count:,} "
        f"({count / len(df) * 100:.2f}%)"
    )

complete_count = int(df["sofa_complete"].sum())

print(
    f"\nComplete total SOFA: {complete_count:,} "
    f"({complete_count / len(df) * 100:.2f}%)"
)


# ============================================================
# 3. SAVE TOTAL SOFA TIMELINE
# ============================================================

print("\nSaving hourly total SOFA...")

TOTAL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    TOTAL_OUTPUT,
    index=False,
)

print("Saved:", TOTAL_OUTPUT)


# ============================================================
# 4. GENERATE FUTURE SOFA OBSERVATIONS
# ============================================================

print("\nGenerating future SOFA values for hours 2-6...")

grouped = df.groupby(
    "stay_id",
    sort=False,
)

future_columns = []

for hour in range(2, 7):

    score_column = f"future_sofa_h{hour}"
    time_column = f"future_time_h{hour}"

    df[score_column] = grouped["total_sofa"].shift(-hour)

    df[time_column] = grouped["assessment_time"].shift(-hour)

    # Confirm that the shifted observation is exactly
    # the intended number of hours in the future.
    actual_delta = (
        df[time_column] - df["assessment_time"]
    )

    valid_time = actual_delta.eq(
        pd.Timedelta(hours=hour)
    )

    df.loc[~valid_time, score_column] = np.nan

    future_columns.append(score_column)

    print(f"Prepared future hour +{hour}")


# ============================================================
# 5. CALCULATE FUTURE DELTA SOFA
# ============================================================

print("\nCalculating future deterioration...")

df["future_complete_count"] = (
    df[future_columns].notna().sum(axis=1)
)

df["future_max_sofa_2_6h"] = (
    df[future_columns].max(axis=1, skipna=True)
)

df["delta_sofa_2_6h"] = (
    df["future_max_sofa_2_6h"] - df["total_sofa"]
)

# Positive:
# At least one valid future complete SOFA assessment
# shows an increase of 2 or more points.

positive = (
    df["sofa_complete"]
    & df["delta_sofa_2_6h"].ge(2)
)

# Negative:
# Baseline complete and all five future assessments
# are complete, with no increase of 2 or more.

negative = (
    df["sofa_complete"]
    & df["future_complete_count"].eq(5)
    & df["delta_sofa_2_6h"].lt(2)
)

df["deterioration_label"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64",
)

df.loc[positive, "deterioration_label"] = 1
df.loc[negative, "deterioration_label"] = 0


# ============================================================
# 6. SAVE LABELED DATASET
# ============================================================

print("\nPreparing final labeled dataset...")

label_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "assessment_time",
    *components,
    "available_component_count",
    "total_sofa",
    "future_max_sofa_2_6h",
    "delta_sofa_2_6h",
    "future_complete_count",
    "deterioration_label",
]

labeled = df[label_columns].copy()

labeled.to_csv(
    LABEL_OUTPUT,
    index=False,
)

print("Saved:", LABEL_OUTPUT)


# ============================================================
# 7. GENERATE VALIDATION REPORT
# ============================================================

print("\nGenerating validation report...")

total_rows = len(df)

positive_count = int(positive.sum())
negative_count = int(negative.sum())

unlabeled_count = int(
    df["deterioration_label"].isna().sum()
)

labeled_count = positive_count + negative_count

report = {
    "dataset": "MIMIC-IV hourly deterioration dataset",
    "cohort": "Strict ICU cohort",
    "total_assessments": int(total_rows),
    "total_icu_stays": int(df["stay_id"].nunique()),
    "complete_total_sofa_assessments": complete_count,
    "complete_total_sofa_percentage": round(
        complete_count / total_rows * 100, 2
    ),
    "prediction_target": "Delta SOFA >= 2",
    "prediction_horizon_hours": [2, 6],
    "baseline_requires_all_six_components": True,
    "negative_label_requires_all_five_future_assessments": True,
    "positive_label_requires_at_least_one_valid_future_deterioration": True,
    "missing_components_imputed_as_zero": False,
    "labeled_assessments": labeled_count,
    "positive_labels": positive_count,
    "negative_labels": negative_count,
    "unlabeled_assessments": unlabeled_count,
    "positive_percentage_among_labeled": round(
        positive_count / labeled_count * 100, 2
    ) if labeled_count else None,
    "negative_percentage_among_labeled": round(
        negative_count / labeled_count * 100, 2
    ) if labeled_count else None,
    "label_coverage_percentage": round(
        labeled_count / total_rows * 100, 2
    ),
    "component_coverage": {
        column: {
            "available": int(df[column].notna().sum()),
            "percentage": round(
                df[column].notna().mean() * 100, 2
            ),
        }
        for column in components
    },
    "clinical_note": (
        "Research deterioration labels based on complete six-component "
        "hourly SOFA. Positive labels require at least one valid future "
        "SOFA increase of >=2 within hours 2-6. Negative labels require "
        "all five future hourly SOFA assessments to be complete. "
        "Uncertain cases remain unlabeled."
    ),
}

with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=4,
    )


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

print("\n========== TOTAL SOFA SUMMARY ==========")

print(f"Total assessments : {total_rows:,}")
print(f"Complete SOFA     : {complete_count:,}")

print("\n========== DETERIORATION LABEL SUMMARY ==========")

print(f"Positive labels   : {positive_count:,}")
print(f"Negative labels   : {negative_count:,}")
print(f"Unlabeled         : {unlabeled_count:,}")
print(f"Label coverage    : {report['label_coverage_percentage']}%")

if labeled_count:

    print(
        "Positive class    : "
        f"{report['positive_percentage_among_labeled']}%"
    )

print("\nSaved:", TOTAL_OUTPUT)
print("Saved:", LABEL_OUTPUT)
print("Saved:", REPORT_OUTPUT)

print("\nTotal SOFA and deterioration-label generation completed.")
