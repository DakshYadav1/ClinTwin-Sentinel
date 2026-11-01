import json
from pathlib import Path
import pandas as pd

SOURCE = Path("data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz")
OUTPUT = Path("data/real/processed/cleaned_vitals.csv")
REPORT = Path("data/real/processed/vitals_cleaning_report.json")

# Item ID: feature name, lower bound, upper bound, original unit
RULES = {
    220045: ("heart_rate", 20, 250, "bpm"),
    220210: ("respiratory_rate", 1, 100, "insp/min"),
    223761: ("temperature", 77, 113, "°F"),
    223762: ("temperature", 25, 45, "°C"),
    220050: ("arterial_systolic_bp", 40, 300, "mmHg"),
    220051: ("arterial_diastolic_bp", 10, 200, "mmHg"),
    220052: ("arterial_mean_bp", 20, 250, "mmHg"),
    220179: ("nibp_systolic_bp", 40, 300, "mmHg"),
    220180: ("nibp_diastolic_bp", 10, 200, "mmHg"),
    220181: ("nibp_mean_bp", 20, 250, "mmHg"),
}

columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "storetime", "itemid",
    "valuenum", "valueuom"
]

report = {
    "source": str(SOURCE),
    "output": str(OUTPUT),
    "retained_total": 0,
    "excluded_missing_numeric": 0,
    "excluded_below_range": 0,
    "excluded_above_range": 0,
    "temperature_fahrenheit_converted": 0,
    "by_item": {}
}

for itemid, (feature, low, high, unit) in RULES.items():
    report["by_item"][str(itemid)] = {
        "feature": feature,
        "screening_range": [low, high],
        "original_unit": unit,
        "retained": 0,
        "excluded_missing": 0,
        "excluded_below": 0,
        "excluded_above": 0
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    OUTPUT.unlink()

first_write = True

for chunk in pd.read_csv(
    SOURCE,
    compression="gzip",
    usecols=columns,
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(RULES)].copy()

    if chunk.empty:
        continue

    output_parts = []

    for itemid, group in chunk.groupby("itemid"):
        itemid = int(itemid)
        feature, low, high, original_unit = RULES[itemid]
        item_report = report["by_item"][str(itemid)]

        missing = group["valuenum"].isna()
        below = group["valuenum"] < low
        above = group["valuenum"] > high

        item_report["excluded_missing"] += int(missing.sum())
        item_report["excluded_below"] += int(below.sum())
        item_report["excluded_above"] += int(above.sum())

        report["excluded_missing_numeric"] += int(missing.sum())
        report["excluded_below_range"] += int(below.sum())
        report["excluded_above_range"] += int(above.sum())

        valid = group.loc[~(missing | below | above)].copy()

        if valid.empty:
            continue

        valid["feature"] = feature
        valid["original_value"] = valid["valuenum"]
        valid["original_unit"] = original_unit
        valid["unit"] = original_unit
        valid["value"] = valid["valuenum"]

        if itemid == 223761:
            valid["value"] = (valid["valuenum"] - 32) * 5 / 9
            valid["unit"] = "°C"
            report["temperature_fahrenheit_converted"] += len(valid)

        valid["charttime"] = pd.to_datetime(valid["charttime"])
        valid["storetime"] = pd.to_datetime(valid["storetime"])

        item_report["retained"] += len(valid)
        report["retained_total"] += len(valid)

        output_parts.append(valid[
            [
                "subject_id", "hadm_id", "stay_id",
                "charttime", "storetime", "itemid",
                "feature", "original_value", "original_unit",
                "value", "unit"
            ]
        ])

    if output_parts:
        result = pd.concat(output_parts, ignore_index=True)
        result.to_csv(
            OUTPUT,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )
        first_write = False

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\n========== VITAL CLEANING REPORT ==========")
print(json.dumps(report, indent=2))
print("\nOutput:", OUTPUT)
print("Audit report:", REPORT)
