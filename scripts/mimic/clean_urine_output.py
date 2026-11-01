import json
from pathlib import Path
import pandas as pd

SOURCE = Path("data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz")
OUTPUT = Path("data/real/processed/cleaned_urine_output.csv")
REPORT = Path("data/real/processed/urine_cleaning_report.json")

CHUNK_SIZE = 1_000_000
URINE_IDS = {226559: "Foley", 226560: "Void"}
IRRIGATION_IDS = {227488, 227489}
INVALID_VOID = 876587.0

columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "storetime", "itemid", "value"
]

print("Step 1/2: Identifying stay-hour periods with irrigation...")

irrigation_hours = set()

for chunk in pd.read_csv(
    SOURCE, compression="gzip",
    usecols=["stay_id", "charttime", "itemid"],
    chunksize=CHUNK_SIZE
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    irrigation = chunk[chunk["itemid"].isin(IRRIGATION_IDS)]

    if not irrigation.empty:
        irrigation["hour"] = pd.to_datetime(
            irrigation["charttime"]
        ).dt.floor("h")

        irrigation_hours.update(
            map(tuple, irrigation[["stay_id", "hour"]]
                .drop_duplicates().to_numpy())
        )

print(f"Identified {len(irrigation_hours):,} stay-hour periods.")

print("\nStep 2/2: Extracting and cleaning Foley/Void records...")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    OUTPUT.unlink()

report = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "source_item_ids": URINE_IDS,
    "irrigation_item_ids": sorted(IRRIGATION_IDS),
    "rules": {
        "negative_values": "Excluded",
        "void_876587_ml": "Excluded as an implausible extreme",
        "values_above_2000_ml": "Retained and flagged",
        "foley_same_hour_as_irrigation": "Retained and flagged",
        "irrigation_and_or_urine": "Not included in routine urine output"
    },
    "total_selected": 0,
    "retained": 0,
    "excluded_negative": 0,
    "excluded_extreme_void": 0,
    "extreme_value_flagged": 0,
    "irrigation_overlap_flagged": 0,
    "by_source": {
        "Foley": {"retained": 0, "excluded": 0},
        "Void": {"retained": 0, "excluded": 0}
    }
}

first_write = True

for chunk in pd.read_csv(
    SOURCE, compression="gzip",
    usecols=columns, chunksize=CHUNK_SIZE
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(URINE_IDS)].copy()

    if chunk.empty:
        continue

    chunk["source"] = chunk["itemid"].map(URINE_IDS)
    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["storetime"] = pd.to_datetime(chunk["storetime"])

    report["total_selected"] += len(chunk)

    negative = chunk["value"] < 0
    invalid_void = (
        (chunk["itemid"] == 226560) &
        (chunk["value"] == INVALID_VOID)
    )

    excluded = negative | invalid_void

    report["excluded_negative"] += int(negative.sum())
    report["excluded_extreme_void"] += int(
        (invalid_void & ~negative).sum()
    )

    for source in URINE_IDS.values():
        source_mask = chunk["source"] == source
        report["by_source"][source]["excluded"] += int(
            (source_mask & excluded).sum()
        )

    valid = chunk.loc[~excluded].copy()

    if valid.empty:
        continue

    valid["hour"] = valid["charttime"].dt.floor("h")

    valid["extreme_value_flag"] = valid["value"] > 2000

    valid["irrigation_overlap_flag"] = False

    foley_mask = valid["itemid"] == 226559
    valid.loc[foley_mask, "irrigation_overlap_flag"] = [
        (stay, hour) in irrigation_hours
        for stay, hour in zip(
            valid.loc[foley_mask, "stay_id"],
            valid.loc[foley_mask, "hour"]
        )
    ]

    valid["quality_flag"] = ""
    valid.loc[
        valid["extreme_value_flag"], "quality_flag"
    ] = "extreme_value"

    overlap = valid["irrigation_overlap_flag"]
    both = overlap & valid["extreme_value_flag"]

    valid.loc[overlap & ~both, "quality_flag"] = "irrigation_overlap"
    valid.loc[both, "quality_flag"] = (
        "extreme_value;irrigation_overlap"
    )

    report["retained"] += len(valid)
    report["extreme_value_flagged"] += int(
        valid["extreme_value_flag"].sum()
    )
    report["irrigation_overlap_flagged"] += int(overlap.sum())

    for source in URINE_IDS.values():
        report["by_source"][source]["retained"] += int(
            (valid["source"] == source).sum()
        )

    valid = valid.drop(columns=["hour"])

    valid.to_csv(
        OUTPUT, mode="w" if first_write else "a",
        header=first_write, index=False
    )
    first_write = False

report["irrigation_stay_hour_periods"] = len(irrigation_hours)

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n========== CLEANING REPORT ==========")
print(json.dumps(report, indent=2))
print("\nOutput:", OUTPUT)
print("Audit report:", REPORT)
