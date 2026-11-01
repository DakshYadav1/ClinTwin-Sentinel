
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CHARTEVENTS_PATH = (
    BASE_DIR / "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"
)

COHORT_PATH = (
    BASE_DIR / "data/real/processed/strict_icu_cohort.csv"
)

SCHEDULE_PATH = (
    BASE_DIR / "data/real/processed/cv_assessment_schedule.csv"
)

OBSERVATIONS_PATH = (
    BASE_DIR / "data/real/processed/gcs_component_observations.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data/real/processed/hourly_neurological_sofa.csv"
)

REPORT_PATH = (
    BASE_DIR / "data/real/processed/neurological_sofa_report.json"
)

GCS_ITEMS = {
    220739: "eye",
    223900: "verbal",
    223901: "motor",
}

# Maximum permitted age difference when pairing GCS components.
PAIRING_WINDOW_NS = 60 * 60 * 1_000_000_000

# Maximum time a complete GCS observation can be carried forward.
CARRY_FORWARD_NS = 6 * 60 * 60 * 1_000_000_000

CHUNK_SIZE = 2_000_000


# ============================================================
# 1. LOAD COHORT
# ============================================================

print("Loading strict ICU cohort...")

cohort = pd.read_csv(
    COHORT_PATH,
    parse_dates=["intime", "outtime"]
)

cohort = cohort[
    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
].dropna().copy()

valid_stays = set(cohort["stay_id"].astype(int))

print(f"Strict ICU stays: {len(valid_stays):,}")


# ============================================================
# 2. EXTRACT GCS OBSERVATIONS FROM CHARTEVENTS
# ============================================================

print("\nExtracting GCS observations from chartevents...")

if OBSERVATIONS_PATH.exists():
    OBSERVATIONS_PATH.unlink()

usecols = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "charttime",
    "storetime",
    "itemid",
    "value",
    "valuenum",
]

chunks = pd.read_csv(
    CHARTEVENTS_PATH,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

rows_written = 0

for chunk_number, chunk in enumerate(chunks, start=1):

    chunk = chunk[
        chunk["itemid"].isin(GCS_ITEMS)
        & chunk["stay_id"].isin(valid_stays)
    ].copy()

    if chunk.empty:
        continue

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"],
        errors="coerce"
    )

    chunk["storetime"] = pd.to_datetime(
        chunk["storetime"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["stay_id", "charttime", "valuenum"]
    )

    if chunk.empty:
        continue

    chunk.to_csv(
        OBSERVATIONS_PATH,
        mode="a",
        header=not OBSERVATIONS_PATH.exists(),
        index=False,
    )

    rows_written += len(chunk)

    print(
        f"Chunk {chunk_number:,}: "
        f"{rows_written:,} GCS observations extracted"
    )

print(f"\nTotal extracted observations: {rows_written:,}")


# ============================================================
# 3. LOAD AND VALIDATE GCS OBSERVATIONS
# ============================================================

print("\nLoading extracted GCS observations...")

gcs = pd.read_csv(
    OBSERVATIONS_PATH,
    parse_dates=["charttime", "storetime"]
)

gcs = gcs.merge(
    cohort,
    on=["subject_id", "hadm_id", "stay_id"],
    how="inner",
    suffixes=("", "_cohort"),
)

gcs = gcs[
    (gcs["charttime"] >= gcs["intime"])
    & (gcs["charttime"] <= gcs["outtime"])
].copy()

gcs["gcs_score"] = pd.to_numeric(
    gcs["valuenum"],
    errors="coerce"
)

gcs["component"] = gcs["itemid"].map(GCS_ITEMS)

# Validate numeric component ranges.
valid_ranges = {
    "eye": (1, 4),
    "verbal": (1, 5),
    "motor": (1, 6),
}

valid = pd.Series(False, index=gcs.index)

for component, (lower, upper) in valid_ranges.items():

    mask = gcs["component"].eq(component)

    valid.loc[mask] = (
        gcs.loc[mask, "gcs_score"].between(lower, upper)
    )

gcs = gcs[valid].copy()

# Preserve the ETT verbal-response flag.
gcs["intubated_verbal"] = (
    gcs["value"]
    .astype("string")
    .str.contains("ETT", case=False, na=False)
)

# If multiple records exist for the same component and timestamp,
# retain the latest recorded entry.
gcs = gcs.sort_values(
    ["stay_id", "charttime", "itemid", "storetime"],
    kind="mergesort"
)

gcs = gcs.drop_duplicates(
    subset=["stay_id", "charttime", "itemid"],
    keep="last"
)

gcs = gcs[
    [
        "stay_id",
        "charttime",
        "component",
        "gcs_score",
        "intubated_verbal",
    ]
].copy()

print(f"Valid GCS component observations: {len(gcs):,}")

print("\nComponent counts:")
print(gcs["component"].value_counts().to_string())


# ============================================================
# 4. PREPARE COMPONENT OBSERVATIONS BY STAY
# ============================================================

print("\nPreparing GCS observations by ICU stay...")

component_groups = {}

for stay_id, group in gcs.groupby("stay_id", sort=False):

    component_groups[stay_id] = {}

    for component in ["eye", "verbal", "motor"]:

        part = group[
            group["component"].eq(component)
        ].sort_values("charttime")

        component_groups[stay_id][component] = {
            "times": part["charttime"].astype("int64").to_numpy(),
            "scores": part["gcs_score"].to_numpy(dtype=np.float64),
            "intubated": part["intubated_verbal"].to_numpy(dtype=bool),
        }


# ============================================================
# 5. PAIR GCS COMPONENTS
# ============================================================

print("\nPairing GCS components within one hour...")

paired_groups = {}

for processed, (stay_id, components) in enumerate(
    component_groups.items(),
    start=1
):

    available_times = [
        components[name]["times"]
        for name in ["eye", "verbal", "motor"]
        if len(components[name]["times"]) > 0
    ]

    if len(available_times) != 3:
        continue

    # Use all observed component timestamps as candidate anchors.
    anchors = np.unique(
        np.concatenate(available_times)
    )

    result = []

    for anchor in anchors:

        selected = {}

        complete = True

        for name in ["eye", "verbal", "motor"]:

            data = components[name]

            index = np.searchsorted(
                data["times"],
                anchor,
                side="right"
            ) - 1

            if index < 0:
                complete = False
                break

            age = anchor - data["times"][index]

            if age < 0 or age > PAIRING_WINDOW_NS:
                complete = False
                break

            selected[name] = index

        if not complete:
            continue

        eye_index = selected["eye"]
        verbal_index = selected["verbal"]
        motor_index = selected["motor"]

        eye_score = components["eye"]["scores"][eye_index]
        verbal_score = components["verbal"]["scores"][verbal_index]
        motor_score = components["motor"]["scores"][motor_index]

        total_gcs = eye_score + verbal_score + motor_score

        if not 3 <= total_gcs <= 15:
            continue

        intubated = components["verbal"]["intubated"][verbal_index]

        result.append(
            (
                anchor,
                eye_score,
                verbal_score,
                motor_score,
                total_gcs,
                intubated,
            )
        )

    if result:
        paired_groups[stay_id] = np.array(
            result,
            dtype=object
        )

    if processed % 10000 == 0:
        print(
            f"Processed {processed:,} / "
            f"{len(component_groups):,} stays"
        )

print(f"Stays with complete paired GCS: {len(paired_groups):,}")


# ============================================================
# 6. LOAD HOURLY ASSESSMENT SCHEDULE
# ============================================================

print("\nLoading hourly assessment schedule...")

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
# 7. ASSIGN LATEST GCS TO HOURLY ASSESSMENTS
# ============================================================

print("\nAssigning GCS to hourly assessments...")

eye_output = np.full(len(schedule), np.nan)
verbal_output = np.full(len(schedule), np.nan)
motor_output = np.full(len(schedule), np.nan)
total_output = np.full(len(schedule), np.nan)

intubated_output = np.zeros(len(schedule), dtype=bool)
gcs_age_minutes = np.full(len(schedule), np.nan)

schedule_groups = schedule.groupby(
    "stay_id",
    sort=False
).indices

for processed, (stay_id, positions) in enumerate(
    schedule_groups.items(),
    start=1
):

    if stay_id not in paired_groups:
        continue

    paired = paired_groups[stay_id]

    paired_times = paired[:, 0].astype(np.int64)

    assessment_times = (
        schedule.loc[positions, "assessment_time"]
        .astype("int64")
        .to_numpy()
    )

    indices = (
        np.searchsorted(
            paired_times,
            assessment_times,
            side="right"
        ) - 1
    )

    for position, assessment_time, index in zip(
        positions,
        assessment_times,
        indices
    ):

        if index < 0:
            continue

        age = assessment_time - paired_times[index]

        if age < 0 or age > CARRY_FORWARD_NS:
            continue

        observation = paired[index]

        eye_output[position] = float(observation[1])
        verbal_output[position] = float(observation[2])
        motor_output[position] = float(observation[3])
        total_output[position] = float(observation[4])

        intubated_output[position] = bool(observation[5])

        gcs_age_minutes[position] = age / (60 * 1_000_000_000)

    if processed % 10000 == 0:
        print(
            f"Processed {processed:,} / "
            f"{len(schedule_groups):,} stays"
        )


# ============================================================
# 8. CALCULATE NEUROLOGICAL SOFA
# ============================================================

print("\nCalculating neurological SOFA...")

schedule["gcs_eye"] = eye_output
schedule["gcs_verbal"] = verbal_output
schedule["gcs_motor"] = motor_output
schedule["total_gcs"] = total_output

schedule["intubated_verbal_flag"] = intubated_output
schedule["gcs_age_minutes"] = gcs_age_minutes

schedule["neurological_sofa"] = np.select(
    [
        schedule["total_gcs"].isna(),
        schedule["total_gcs"] == 15,
        schedule["total_gcs"].between(13, 14),
        schedule["total_gcs"].between(10, 12),
        schedule["total_gcs"].between(6, 9),
        schedule["total_gcs"].between(3, 5),
    ],
    [
        np.nan,
        0,
        1,
        2,
        3,
        4,
    ],
    default=np.nan
)

schedule["neurological_sofa"] = pd.array(
    schedule["neurological_sofa"],
    dtype="Int64"
)


# ============================================================
# 9. SAVE OUTPUT
# ============================================================

print("\nSaving neurological SOFA...")

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

schedule.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 10. GENERATE REPORT
# ============================================================

total = len(schedule)

available = int(
    schedule["neurological_sofa"].notna().sum()
)

missing = total - available

distribution = {
    str(int(score)): int(count)
    for score, count in
    schedule["neurological_sofa"].value_counts().sort_index().items()
}

report = {
    "component": "Neurological SOFA",
    "gcs_item_ids": GCS_ITEMS,
    "gcs_pairing_window_minutes": 60,
    "carry_forward_hours": 6,
    "total_icu_stays": int(schedule["stay_id"].nunique()),
    "total_assessments": int(total),
    "available_scores": available,
    "missing_scores": missing,
    "coverage_percentage": round(
        available / total * 100, 2
    ) if total else 0,
    "intubated_verbal_assessments": int(
        schedule["intubated_verbal_flag"].sum()
    ),
    "score_distribution": distribution,
    "missing_values_treated_as_zero": False,
    "clinical_note": (
        "Hourly research proxy based on paired GCS components. "
        "ETT verbal responses are retained as recorded with score 1 "
        "and separately flagged. GCS pairing and carry-forward "
        "windows are methodological choices, not standard SOFA rules."
    )
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n========== NEUROLOGICAL SOFA SUMMARY ==========")

print(f"Total assessments : {total:,}")
print(f"Available scores  : {available:,}")
print(f"Missing scores    : {missing:,}")
print(f"Coverage          : {report['coverage_percentage']}%")

print(
    "Intubated verbal assessments:",
    f"{report['intubated_verbal_assessments']:,}"
)

print("\nScore distribution:")

for score, count in distribution.items():
    print(f"SOFA {score}: {count:,}")

print("\nSample output:")

print(
    schedule[
        [
            "stay_id",
            "assessment_time",
            "gcs_eye",
            "gcs_verbal",
            "gcs_motor",
            "total_gcs",
            "intubated_verbal_flag",
            "neurological_sofa",
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:", OUTPUT_PATH)
print("Saved:", REPORT_PATH)

print("\nNeurological SOFA processing completed.")
