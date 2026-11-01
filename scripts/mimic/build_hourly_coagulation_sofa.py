import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

COHORT_PATH = (
    BASE_DIR
    / "data/real/processed/strict_icu_cohort.csv"
)

LABS_PATH = (
    BASE_DIR
    / "data/real/processed/cleaned_labevents.csv"
)

SCHEDULE_PATH = (
    BASE_DIR
    / "data/real/processed/cv_assessment_schedule.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data/real/processed/hourly_coagulation_sofa.csv"
)

REPORT_PATH = (
    BASE_DIR
    / "data/real/processed/coagulation_sofa_report.json"
)

PLATELET_ITEM_IDS = [51265, 53189]

WINDOW_NS = 24 * 60 * 60 * 1_000_000_000


# ============================================================
# 1. LOAD STRICT ICU COHORT
# ============================================================

print("Loading strict ICU cohort...")

cohort = pd.read_csv(
    COHORT_PATH,
    parse_dates=["intime", "outtime"]
)

cohort = cohort[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
    ]
].copy()

cohort = cohort.dropna(
    subset=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
)

print(f"Strict ICU stays: {len(cohort):,}")


# ============================================================
# 2. LOAD AND FILTER PLATELET OBSERVATIONS
# ============================================================

print("Loading platelet observations...")

labs = pd.read_csv(
    LABS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    parse_dates=["charttime"],
)

labs = labs[
    labs["itemid"].isin(PLATELET_ITEM_IDS)
].copy()

labs = labs.dropna(
    subset=["subject_id", "hadm_id", "charttime", "valuenum"]
)

# Platelet counts in MIMIC-IV are expected in K/uL.
labs = labs[
    labs["valueuom"].eq("K/uL")
].copy()

labs = labs[
    (labs["valuenum"] > 0)
    & (labs["valuenum"] <= 2000)
].copy()

labs = labs.rename(
    columns={
        "charttime": "event_time",
        "valuenum": "platelet",
    }
)

# Keep only the fields needed for cohort linkage.
labs = labs[
    [
        "subject_id",
        "hadm_id",
        "event_time",
        "platelet",
    ]
]

print(f"Platelet observations before ICU linkage: {len(labs):,}")


# ============================================================
# 3. LINK PLATELETS TO STRICT ICU STAYS
# ============================================================

print("Linking platelet observations to ICU stays...")

labs = labs.merge(
    cohort,
    on=["subject_id", "hadm_id"],
    how="inner",
)

# Retain observations occurring during the corresponding ICU stay.
labs = labs[
    (labs["event_time"] >= labs["intime"])
    & (labs["event_time"] <= labs["outtime"])
].copy()

labs = labs[
    [
        "subject_id",
        "hadm_id",
        "stay_id",
        "event_time",
        "platelet",
    ]
].copy()

labs["platelet"] = pd.to_numeric(
    labs["platelet"],
    errors="coerce",
)

labs = labs.dropna(
    subset=["stay_id", "event_time", "platelet"]
)

labs = labs.sort_values(
    ["stay_id", "event_time"],
    kind="mergesort",
).reset_index(drop=True)

print(
    "Platelet observations in strict ICU stays:",
    f"{len(labs):,}",
)


# ============================================================
# 4. LOAD HOURLY ASSESSMENT SCHEDULE
# ============================================================

print("Loading hourly assessment schedule...")

schedule = pd.read_csv(
    SCHEDULE_PATH,
    parse_dates=["assessment_time"],
)

schedule = schedule.dropna(
    subset=["stay_id", "assessment_time"]
).copy()

schedule = schedule.sort_values(
    ["stay_id", "assessment_time"],
    kind="mergesort",
).reset_index(drop=True)

print(f"Hourly assessments: {len(schedule):,}")


# ============================================================
# 5. PREPARE PLATELET OBSERVATIONS BY ICU STAY
# ============================================================

print("Preparing platelet observations by ICU stay...")

lab_groups = {}

for stay_id, group in labs.groupby("stay_id", sort=False):

    times = group["event_time"].astype("int64").to_numpy()

    values = group["platelet"].to_numpy(dtype=np.float64)

    lab_groups[stay_id] = (times, values)


# ============================================================
# 6. CALCULATE ROLLING 24-HOUR MINIMUM
# ============================================================

print("Calculating rolling 24-hour platelet minimum...")

# Each schedule row receives its own result.
platelet_minimum = np.full(
    len(schedule),
    np.nan,
    dtype=np.float64,
)

# Process one ICU stay at a time.
schedule_groups = schedule.groupby(
    "stay_id",
    sort=False,
).indices

for processed, (stay_id, row_positions) in enumerate(
    schedule_groups.items(),
    start=1,
):

    if stay_id not in lab_groups:
        continue

    lab_times, lab_values = lab_groups[stay_id]

    assessment_times = (
        schedule.loc[row_positions, "assessment_time"]
        .astype("int64")
        .to_numpy()
    )

    # Monotonic deque stores indices of candidate minimum values.
    # Values in the deque are maintained in increasing order.
    min_queue = deque()

    lab_pointer = 0
    n_labs = len(lab_times)

    for position, assessment_time in zip(
        row_positions,
        assessment_times,
    ):

        window_start = assessment_time - WINDOW_NS

        # Add every lab observation available at or before
        # the current assessment time.
        while (
            lab_pointer < n_labs
            and lab_times[lab_pointer] <= assessment_time
        ):

            current_value = lab_values[lab_pointer]

            # Remove larger values from the back.
            while (
                min_queue
                and lab_values[min_queue[-1]] >= current_value
            ):
                min_queue.pop()

            min_queue.append(lab_pointer)

            lab_pointer += 1

        # Remove observations older than the 24-hour window.
        # The boundary is inclusive: exactly 24h old is retained.
        while (
            min_queue
            and lab_times[min_queue[0]] < window_start
        ):
            min_queue.popleft()

        # If at least one valid observation exists, record its minimum.
        if min_queue:
            platelet_minimum[position] = (
                lab_values[min_queue[0]]
            )

    if processed % 10000 == 0:
        print(
            f"Processed {processed:,} / "
            f"{len(schedule_groups):,} ICU stays"
        )


# ============================================================
# 7. ASSIGN PLATELET MINIMUM AND CALCULATE SOFA SCORE
# ============================================================

print("Calculating coagulation SOFA scores...")

schedule["platelet_min_24h"] = platelet_minimum

# SOFA coagulation component:
#
# Platelets >= 150       -> 0
# Platelets >= 100       -> 1
# Platelets >= 50        -> 2
# Platelets >= 20        -> 3
# Platelets < 20         -> 4
#
# Missing platelet data remains missing.

schedule["coagulation_sofa"] = np.select(
    [
        schedule["platelet_min_24h"].isna(),
        schedule["platelet_min_24h"] >= 150,
        schedule["platelet_min_24h"] >= 100,
        schedule["platelet_min_24h"] >= 50,
        schedule["platelet_min_24h"] >= 20,
        schedule["platelet_min_24h"] < 20,
    ],
    [
        np.nan,
        0,
        1,
        2,
        3,
        4,
    ],
    default=np.nan,
)

# Preserve missing values as NaN rather than assigning zero.
schedule["coagulation_sofa"] = pd.array(
    schedule["coagulation_sofa"],
    dtype="Int64",
)


# ============================================================
# 8. SAVE HOURLY COAGULATION SOFA
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

schedule.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nHourly coagulation SOFA saved:")
print(OUTPUT_PATH)


# ============================================================
# 9. GENERATE VALIDATION REPORT
# ============================================================

total_assessments = len(schedule)

available = int(
    schedule["coagulation_sofa"].notna().sum()
)

missing = total_assessments - available

score_distribution = (
    schedule["coagulation_sofa"]
    .value_counts()
    .sort_index()
    .to_dict()
)

score_distribution = {
    str(int(score)): int(count)
    for score, count in score_distribution.items()
}

report = {
    "component": "Coagulation SOFA",
    "cohort": "Strict ICU cohort",
    "platelet_item_ids": PLATELET_ITEM_IDS,
    "platelet_unit": "K/uL",
    "lookback_window_hours": 24,
    "window_boundary": "Inclusive",
    "total_icu_stays": int(schedule["stay_id"].nunique()),
    "total_hourly_assessments": int(total_assessments),
    "assessments_with_platelet_score": available,
    "assessments_missing_platelet_score": missing,
    "coverage_percentage": round(
        (available / total_assessments * 100)
        if total_assessments
        else 0,
        2,
    ),
    "score_distribution": score_distribution,
    "missing_values_treated_as_zero": False,
    "clinical_note": (
        "Hourly research proxy using the minimum platelet count "
        "in the preceding 24 hours. This is not a claim that "
        "hourly scoring is equivalent to standard clinical SOFA."
    ),
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=4,
    )

print("\nCoagulation SOFA report saved:")
print(REPORT_PATH)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n========== COAGULATION SOFA SUMMARY ==========")

print(f"Total assessments : {total_assessments:,}")
print(f"Available scores  : {available:,}")
print(f"Missing scores    : {missing:,}")

print(
    f"Coverage          : "
    f"{report['coverage_percentage']}%"
)

print("\nScore distribution:")

for score, count in score_distribution.items():
    print(f"SOFA {score}: {count:,}")

print("\nSample output:")

print(
    schedule[
        [
            "stay_id",
            "assessment_time",
            "platelet_min_24h",
            "coagulation_sofa",
        ]
    ].head(10).to_string(index=False)
)

print("\nCoagulation SOFA processing completed.")