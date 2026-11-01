
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

COHORT_PATH = BASE_DIR / "data/real/processed/strict_icu_cohort.csv"
LABS_PATH = BASE_DIR / "data/real/processed/cleaned_labevents.csv"
URINE_PATH = BASE_DIR / "data/real/processed/cleaned_urine_output.csv"
SCHEDULE_PATH = BASE_DIR / "data/real/processed/cv_assessment_schedule.csv"

OUTPUT_PATH = BASE_DIR / "data/real/processed/hourly_renal_sofa.csv"
REPORT_PATH = BASE_DIR / "data/real/processed/renal_sofa_report.json"

CREATININE_ITEM_IDS = [50912, 52546]

WINDOW_NS = 24 * 60 * 60 * 1_000_000_000

# Research data-quality requirement, not a clinical SOFA threshold.
MIN_URINE_OBSERVATION_HOURS = 18


# ============================================================
# 1. LOAD COHORT AND ASSESSMENT SCHEDULE
# ============================================================

print("Loading strict ICU cohort...")

cohort = pd.read_csv(
    COHORT_PATH,
    parse_dates=["intime", "outtime"]
)

cohort = cohort[
    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
].dropna().copy()


print("Loading hourly assessment schedule...")

schedule = pd.read_csv(
    SCHEDULE_PATH,
    parse_dates=["assessment_time"]
)

schedule = schedule.dropna(
    subset=["stay_id", "assessment_time"]
).sort_values(
    ["stay_id", "assessment_time"],
    kind="mergesort"
).reset_index(drop=True)

print(f"Hourly assessments: {len(schedule):,}")


# ============================================================
# 2. LOAD AND PROCESS CREATININE
# ============================================================

print("\nLoading creatinine observations...")

labs = pd.read_csv(
    LABS_PATH,
    usecols=[
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom"
    ],
    parse_dates=["charttime"]
)

labs = labs[
    labs["itemid"].isin(CREATININE_ITEM_IDS)
].copy()

labs = labs.dropna(
    subset=["subject_id", "hadm_id", "charttime", "valuenum"]
)

units = (
    labs["valueuom"]
    .astype("string")
    .str.lower()
    .str.replace(" ", "", regex=False)
)

mg_dl = units.isin(["mg/dl", "mg/dl."])

umol_l = units.isin([
    "umol/l",
    "µmol/l",
    "μmol/l"
])

labs.loc[umol_l, "valuenum"] = (
    labs.loc[umol_l, "valuenum"] / 88.4
)

labs = labs[mg_dl | umol_l].copy()

labs = labs[
    (labs["valuenum"] > 0)
    & (labs["valuenum"] <= 50)
].copy()

labs = labs.rename(
    columns={
        "charttime": "event_time",
        "valuenum": "creatinine"
    }
)

print("Linking creatinine to ICU stays...")

labs = labs.merge(
    cohort,
    on=["subject_id", "hadm_id"],
    how="inner"
)

labs = labs[
    (labs["event_time"] >= labs["intime"])
    & (labs["event_time"] <= labs["outtime"])
].copy()

labs = labs[
    ["stay_id", "event_time", "creatinine"]
].sort_values(
    ["stay_id", "event_time"],
    kind="mergesort"
).reset_index(drop=True)

print(f"Creatinine observations in ICU: {len(labs):,}")


# ============================================================
# 3. LOAD AND PROCESS URINE OUTPUT
# ============================================================

print("\nLoading cleaned urine output...")

urine = pd.read_csv(
    URINE_PATH,
    usecols=[
        "stay_id",
        "charttime",
        "value",
        "source",
        "extreme_value_flag",
        "irrigation_overlap_flag"
    ],
    parse_dates=["charttime"]
)

urine = urine.dropna(
    subset=["stay_id", "charttime", "value"]
).copy()

# Irrigation-overlap records are excluded.
urine = urine[
    ~urine["irrigation_overlap_flag"].fillna(False)
].copy()

# Retain nonnegative output values.
urine = urine[
    (urine["value"] >= 0)
    & (urine["value"] <= 10000)
].copy()

extreme_urine_count = int(
    urine["extreme_value_flag"].fillna(False).sum()
)

# Aggregate records at the same stay and charting timestamp.
# Values are in mL according to the cleaned urine dataset.
urine = (
    urine.groupby(
        ["stay_id", "charttime"],
        as_index=False
    )["value"]
    .sum()
)

urine = urine.rename(
    columns={
        "charttime": "event_time",
        "value": "urine_ml"
    }
)

urine = urine.sort_values(
    ["stay_id", "event_time"],
    kind="mergesort"
).reset_index(drop=True)

print(f"Urine observations after cleaning: {len(urine):,}")
print(f"Flagged extreme records retained: {extreme_urine_count:,}")


# ============================================================
# 4. PREPARE OBSERVATIONS BY ICU STAY
# ============================================================

print("\nPreparing observations by ICU stay...")

creatinine_groups = {}

for stay_id, group in labs.groupby("stay_id", sort=False):

    times = group["event_time"].astype("int64").to_numpy()

    values = group["creatinine"].to_numpy(dtype=np.float64)

    creatinine_groups[stay_id] = (times, values)


urine_groups = {}

for stay_id, group in urine.groupby("stay_id", sort=False):

    times = group["event_time"].astype("int64").to_numpy()

    values = group["urine_ml"].to_numpy(dtype=np.float64)

    urine_groups[stay_id] = (times, values)


# ============================================================
# 5. CALCULATE ROLLING 24-HOUR CREATININE MAXIMUM
# ============================================================

print("\nCalculating rolling 24-hour creatinine maximum...")

creatinine_maximum = np.full(
    len(schedule),
    np.nan,
    dtype=np.float64
)

schedule_groups = schedule.groupby(
    "stay_id",
    sort=False
).indices

for processed, (stay_id, positions) in enumerate(
    schedule_groups.items(),
    start=1
):

    if stay_id not in creatinine_groups:
        continue

    event_times, event_values = creatinine_groups[stay_id]

    assessment_times = (
        schedule.loc[positions, "assessment_time"]
        .astype("int64")
        .to_numpy()
    )

    max_queue = deque()
    pointer = 0

    for position, assessment_time in zip(
        positions,
        assessment_times
    ):

        window_start = assessment_time - WINDOW_NS

        while (
            pointer < len(event_times)
            and event_times[pointer] <= assessment_time
        ):

            value = event_values[pointer]

            while (
                max_queue
                and event_values[max_queue[-1]] <= value
            ):
                max_queue.pop()

            max_queue.append(pointer)
            pointer += 1

        while (
            max_queue
            and event_times[max_queue[0]] < window_start
        ):
            max_queue.popleft()

        if max_queue:
            creatinine_maximum[position] = (
                event_values[max_queue[0]]
            )

    if processed % 10000 == 0:
        print(f"Processed {processed:,} / {len(schedule_groups):,} stays")


# ============================================================
# 6. CALCULATE ROLLING 24-HOUR URINE OUTPUT
# ============================================================

print("\nCalculating rolling 24-hour urine output...")

urine_total = np.full(
    len(schedule),
    np.nan,
    dtype=np.float64
)

urine_observation_hours = np.zeros(
    len(schedule),
    dtype=np.int16
)

for processed, (stay_id, positions) in enumerate(
    schedule_groups.items(),
    start=1
):

    if stay_id not in urine_groups:
        continue

    event_times, event_values = urine_groups[stay_id]

    assessment_times = (
        schedule.loc[positions, "assessment_time"]
        .astype("int64")
        .to_numpy()
    )

    pointer = 0
    window_sum = 0.0
    active_events = deque()

    for position, assessment_time in zip(
        positions,
        assessment_times
    ):

        window_start = assessment_time - WINDOW_NS

        while (
            pointer < len(event_times)
            and event_times[pointer] <= assessment_time
        ):

            value = event_values[pointer]

            active_events.append((event_times[pointer], value))
            window_sum += value

            pointer += 1

        while (
            active_events
            and active_events[0][0] < window_start
        ):

            _, old_value = active_events.popleft()
            window_sum -= old_value

        observed_hours = len(active_events)

        urine_observation_hours[position] = min(
            observed_hours, 32767
        )

        if observed_hours >= MIN_URINE_OBSERVATION_HOURS:
            urine_total[position] = max(0.0, window_sum)

    if processed % 10000 == 0:
        print(f"Processed {processed:,} / {len(schedule_groups):,} stays")


# ============================================================
# 7. CALCULATE CREATININE AND URINE SOFA SCORES
# ============================================================

print("\nCalculating renal SOFA scores...")

schedule["creatinine_max_24h"] = creatinine_maximum

schedule["urine_output_24h_ml"] = urine_total

schedule["urine_observation_hours"] = urine_observation_hours


# Creatinine SOFA:
# <1.2       = 0
# 1.2-1.9    = 1
# 2.0-3.4    = 2
# 3.5-4.9    = 3
# >=5.0      = 4

schedule["creatinine_sofa"] = np.select(
    [
        schedule["creatinine_max_24h"].isna(),
        schedule["creatinine_max_24h"] < 1.2,
        schedule["creatinine_max_24h"] < 2.0,
        schedule["creatinine_max_24h"] < 3.5,
        schedule["creatinine_max_24h"] < 5.0,
        schedule["creatinine_max_24h"] >= 5.0
    ],
    [np.nan, 0, 1, 2, 3, 4],
    default=np.nan
)


# Urine output SOFA:
# >=500 mL  = 0
# 200-499   = 3
# <200      = 4
#
# A urine score is assigned only when the configured
# observation coverage requirement is met.

schedule["urine_sofa"] = np.select(
    [
        schedule["urine_output_24h_ml"].isna(),
        schedule["urine_output_24h_ml"] < 200,
        schedule["urine_output_24h_ml"] < 500,
        schedule["urine_output_24h_ml"] >= 500
    ],
    [np.nan, 4, 3, 0],
    default=np.nan
)


# Take the higher available component score.
creatinine_score = schedule["creatinine_sofa"].to_numpy(
    dtype=np.float64
)

urine_score = schedule["urine_sofa"].to_numpy(
    dtype=np.float64
)

both_missing = (
    np.isnan(creatinine_score)
    & np.isnan(urine_score)
)

renal_score = np.fmax(
    creatinine_score,
    urine_score
)

renal_score[both_missing] = np.nan

schedule["renal_sofa"] = pd.array(
    renal_score,
    dtype="Int64"
)

schedule["creatinine_sofa"] = pd.array(
    schedule["creatinine_sofa"],
    dtype="Int64"
)

schedule["urine_sofa"] = pd.array(
    schedule["urine_sofa"],
    dtype="Int64"
)


# ============================================================
# 8. SAVE OUTPUT
# ============================================================

print("\nSaving hourly renal SOFA...")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

schedule.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 9. GENERATE REPORT
# ============================================================

total = len(schedule)

available = int(schedule["renal_sofa"].notna().sum())

missing = total - available

distribution = {
    str(int(score)): int(count)
    for score, count in
    schedule["renal_sofa"].value_counts().sort_index().items()
}

report = {
    "component": "Renal SOFA",
    "creatinine_item_ids": CREATININE_ITEM_IDS,
    "creatinine_unit": "mg/dL",
    "urine_output_unit": "mL",
    "lookback_window_hours": 24,
    "window_boundary": "Inclusive",
    "minimum_urine_observation_hours": MIN_URINE_OBSERVATION_HOURS,
    "total_icu_stays": int(schedule["stay_id"].nunique()),
    "total_assessments": int(total),
    "available_scores": available,
    "missing_scores": missing,
    "coverage_percentage": round(
        available / total * 100, 2
    ) if total else 0,
    "creatinine_score_available": int(
        schedule["creatinine_sofa"].notna().sum()
    ),
    "urine_score_available": int(
        schedule["urine_sofa"].notna().sum()
    ),
    "score_distribution": distribution,
    "missing_values_treated_as_zero": False,
    "clinical_note": (
        "Hourly research proxy combining maximum creatinine "
        "and 24-hour urine output. Urine-based scoring requires "
        "at least the configured number of distinct charting "
        "timestamps. This is not standard clinical SOFA aggregation."
    )
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(report, f, indent=4)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("\n========== RENAL SOFA SUMMARY ==========")

print(f"Total assessments       : {total:,}")
print(f"Available renal scores  : {available:,}")
print(f"Missing renal scores    : {missing:,}")
print(f"Coverage                : {report['coverage_percentage']}%")

print(
    "\nCreatinine scores available:",
    f"{report['creatinine_score_available']:,}"
)

print(
    "Urine scores available:",
    f"{report['urine_score_available']:,}"
)

print("\nRenal SOFA distribution:")

for score, count in distribution.items():
    print(f"SOFA {score}: {count:,}")

print("\nSample output:")

print(
    schedule[
        [
            "stay_id",
            "assessment_time",
            "creatinine_max_24h",
            "urine_output_24h_ml",
            "urine_observation_hours",
            "creatinine_sofa",
            "urine_sofa",
            "renal_sofa"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
print("Saved:", REPORT_PATH)
print("\nRenal SOFA processing completed.")
