import pandas as pd
import json
from pathlib import Path

SOURCE = Path("data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz")
OUT = Path("data/real/processed")
OUT.mkdir(parents=True, exist_ok=True)

ITEMS = {
    226559: "foley",
    226560: "void"
}

chunksize = 1_000_000
hourly_parts = []

report = {
    "source": str(SOURCE),
    "selected_itemids": ITEMS,
    "rows_scanned": 0,
    "selected_rows": 0,
    "missing_values": 0,
    "negative_values": 0,
    "valid_rows": 0
}

for chunk in pd.read_csv(
    SOURCE,
    compression="gzip",
    usecols=[
        "subject_id", "hadm_id", "stay_id",
        "charttime", "itemid", "value", "valueuom"
    ],
    chunksize=chunksize
):
    report["rows_scanned"] += len(chunk)

    chunk = chunk[chunk["itemid"].isin(ITEMS)].copy()
    report["selected_rows"] += len(chunk)

    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    report["missing_values"] += int(chunk["value"].isna().sum())
    report["negative_values"] += int((chunk["value"] < 0).sum())

    chunk = chunk[
        chunk["value"].notna() &
        (chunk["value"] >= 0) &
        (chunk["valueuom"] == "mL")
    ].copy()

    report["valid_rows"] += len(chunk)

    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["hour"] = chunk["charttime"].dt.floor("h")
    chunk["source"] = chunk["itemid"].map(ITEMS)

    grouped = chunk.groupby(
        ["subject_id", "hadm_id", "stay_id", "hour", "source"],
        as_index=False
    ).agg(
        volume_ml=("value", "sum"),
        measurement_count=("value", "size")
    )

    hourly_parts.append(grouped)

    print(
        f"Scanned {report['rows_scanned']:,} rows | "
        f"Valid urine records: {report['valid_rows']:,}"
    )

hourly = pd.concat(hourly_parts, ignore_index=True)

hourly = hourly.pivot_table(
    index=["subject_id", "hadm_id", "stay_id", "hour"],
    columns="source",
    values="volume_ml",
    aggfunc="sum"
).reset_index()

hourly.columns.name = None

for col in ["foley", "void"]:
    if col not in hourly.columns:
        hourly[col] = float("nan")

hourly = hourly.rename(columns={
    "foley": "foley_output_ml",
    "void": "void_output_ml"
})

hourly["total_recorded_urine_ml"] = (
    hourly["foley_output_ml"].fillna(0) +
    hourly["void_output_ml"].fillna(0)
)

output_path = OUT / "hourly_urine_output.csv"
report_path = OUT / "urine_output_extraction_report.json"

hourly.to_csv(output_path, index=False)

report["hourly_groups"] = len(hourly)
report["output_file"] = str(output_path)
report["aggregation"] = "Sum valid measurements by ICU stay, hour and source"
report["negative_value_policy"] = "Excluded from processed data; raw data preserved"

with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print("\nEXTRACTION COMPLETE")
print(json.dumps(report, indent=4))
print("\nOUTPUT PREVIEW:")
print(hourly.head(10).to_string(index=False))
