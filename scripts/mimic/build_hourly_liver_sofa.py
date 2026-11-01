
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

COHORT_PATH = BASE_DIR / "data/real/processed/strict_icu_cohort.csv"
LABS_PATH = BASE_DIR / "data/real/processed/cleaned_labevents.csv"
SCHEDULE_PATH = BASE_DIR / "data/real/processed/cv_assessment_schedule.csv"

OUTPUT_PATH = BASE_DIR / "data/real/processed/hourly_liver_sofa.csv"
REPORT_PATH = BASE_DIR / "data/real/processed/liver_sofa_report.json"

# MIMIC-IV total bilirubin item IDs
BILIRUBIN_ITEM_IDS = [50885, 53089]

WINDOW_NS = 24 * 60 * 60 * 1_000_000_000


print("Loading strict ICU cohort...")

cohort = pd.read_csv(
    COHORT_PATH,
    parse_dates=["intime", "outtime"]
)

cohort = cohort[
    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
].dropna().copy()


print("Loading total bilirubin observations...")

labs = pd.read_csv(
    LABS_PATH,
    usecols=[
        "subject_id", "hadm_id", "itemid",
        "charttime", "valuenum", "valueuom"
    ],
    parse_dates=["charttime"]
)

labs = labs[labs["itemid"].isin(BILIRUBIN_ITEM_IDS)].copy()
labs = labs.dropna(
    subset=["subject_id", "hadm_id", "charttime", "valuenum"]
)

# Normalize units to mg/dL.
units = (
    labs["valueuom"]
    .astype("string")
    .str.lower()
    .str.replace(" ", "", regex=False)
)

mg_dl = units.isin(["mg/dl", "mg/dl."])
umol_l = units.isin(["umol/l", "µmol/l", "μmol/l"])

labs.loc[umol_l, "valuenum"] = (
    labs.loc[umol_l, "valuenum"] / 17.104
)

labs = labs[mg_dl | umol_l].copy()

# Retain positive bilirubin measurements.
labs = labs[labs["valuenum"] > 0].copy()

labs = labs.rename(
    columns={
        "charttime": "event_time",
        "valuenum": "bilirubin"
    }
)

print("Linking bilirubin observations to ICU stays...")

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
    ["stay_id", "event_time", "bilirubin"]
].sort_values(
    ["stay_id", "event_time"],
    kind="mergesort"
).reset_index(drop=True)

print(f"Valid ICU bilirubin observations: {len(labs):,}")


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


print("Preparing observations by ICU stay...")

lab_groups = {}

for stay_id, group in labs.groupby("stay_id", sort=False):
    times = group["event_time"].astype("int64").to_numpy()
    values = group["bilirubin"].to_numpy(dtype=np.float64)
    lab_groups[stay_id] = (times, values)


print("Calculating rolling 24-hour maximum bilirubin...")

bilirubin_maximum = np.full(len(schedule), np.nan)

schedule_groups = schedule.groupby(
    "stay_id", sort=False
).indices

for processed, (stay_id, row_positions) in enumerate(
    schedule_groups.items(), start=1
):
    if stay_id not in lab_groups:
        continue

    lab_times, lab_values = lab_groups[stay_id]

    assessment_times = (
        schedule.loc[row_positions, "assessment_time"]
        .astype("int64")
        .to_numpy()
    )

    # Monotonic decreasing deque: maximum remains at the front.
    max_queue = deque()
    lab_pointer = 0
    n_labs = len(lab_times)

    for position, assessment_time in zip(
        row_positions, assessment_times
    ):
        window_start = assessment_time - WINDOW_NS

        while (
            lab_pointer < n_labs
            and lab_times[lab_pointer] <= assessment_time
        ):
            current_value = lab_values[lab_pointer]

            while (
                max_queue
                and lab_values[max_queue[-1]] <= current_value
            ):
                max_queue.pop()

            max_queue.append(lab_pointer)
            lab_pointer += 1

        while (
            max_queue
            and lab_times[max_queue[0]] < window_start
        ):
            max_queue.popleft()

        if max_queue:
            bilirubin_maximum[position] = (
                lab_values[max_queue[0]]
            )

    if processed % 10000 == 0:
        print(f"Processed {processed:,} / {len(schedule_groups):,} stays")


print("Calculating liver SOFA scores...")

schedule["bilirubin_max_24h"] = bilirubin_maximum

# SOFA liver thresholds (bilirubin in mg/dL):
# <1.2       -> 0
# 1.2-1.9    -> 1
# 2.0-5.9    -> 2
# 6.0-11.9   -> 3
# >=12       -> 4

schedule["liver_sofa"] = np.select(
    [
        schedule["bilirubin_max_24h"].isna(),
        schedule["bilirubin_max_24h"] < 1.2,
        schedule["bilirubin_max_24h"] < 2.0,
        schedule["bilirubin_max_24h"] < 6.0,
        schedule["bilirubin_max_24h"] < 12.0,
        schedule["bilirubin_max_24h"] >= 12.0,
    ],
    [np.nan, 0, 1, 2, 3, 4],
    default=np.nan
)

schedule["liver_sofa"] = pd.array(
    schedule["liver_sofa"], dtype="Int64"
)


print("Saving hourly liver SOFA...")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
schedule.to_csv(OUTPUT_PATH, index=False)

available = int(schedule["liver_sofa"].notna().sum())
missing = len(schedule) - available

distribution = {
    str(int(score)): int(count)
    for score, count in schedule["liver_sofa"]
    .value_counts()
    .sort_index()
    .items()
}

report = {
    "component": "Liver SOFA",
    "bilirubin_item_ids": BILIRUBIN_ITEM_IDS,
    "unit": "mg/dL",
    "lookback_window_hours": 24,
    "window_boundary": "Inclusive",
    "total_icu_stays": int(schedule["stay_id"].nunique()),
    "total_assessments": int(len(schedule)),
    "available_scores": available,
    "missing_scores": missing,
    "coverage_percentage": round(
        available / len(schedule) * 100, 2
    ) if len(schedule) else 0,
    "score_distribution": distribution,
    "missing_values_treated_as_zero": False,
    "clinical_note": (
        "Hourly research proxy using maximum total bilirubin "
        "in the preceding 24 hours; not standard clinical "
        "24-hour SOFA aggregation."
    )
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

print("\n========== LIVER SOFA SUMMARY ==========")
print(f"Total assessments : {len(schedule):,}")
print(f"Available scores  : {available:,}")
print(f"Missing scores    : {missing:,}")
print(f"Coverage          : {report['coverage_percentage']}%")
print("\nScore distribution:")

for score, count in distribution.items():
    print(f"SOFA {score}: {count:,}")

print("\nSample:")
print(
    schedule[
        ["stay_id", "assessment_time",
         "bilirubin_max_24h", "liver_sofa"]
    ].head(10).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
print("Saved:", REPORT_PATH)
print("Liver SOFA processing completed.")
