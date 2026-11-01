from pathlib import Path
import pandas as pd

FILTERED = Path("data/real/processed/filtered_labevents.csv")
RAW = Path("data/real/raw/mimiciv-3.1/hosp/labevents.csv.gz")
OUTPUT = Path("data/real/processed/suspicious_lab_source_records.csv")

TARGETS = {
    50813: 20,
    50885: 50,
    50912: 30,
    51265: 1000,
    51301: 500
}

filtered = pd.read_csv(
    FILTERED,
    usecols=["labevent_id", "itemid", "valuenum"]
)

mask = pd.Series(False, index=filtered.index)

for item_id, threshold in TARGETS.items():
    mask |= (
        (filtered["itemid"] == item_id) &
        (filtered["valuenum"] > threshold)
    )

target_ids = set(
    filtered.loc[mask, "labevent_id"].dropna().astype("int64")
)

print(f"Records to inspect: {len(target_ids):,}")
print("Scanning original MIMIC laboratory file...")

results = []

columns = [
    "labevent_id", "itemid", "value", "valuenum",
    "valueuom", "ref_range_lower", "ref_range_upper",
    "flag", "comments"
]

for chunk in pd.read_csv(
    RAW,
    usecols=columns,
    chunksize=500_000,
    compression="gzip",
    low_memory=False
):
    matches = chunk[
        chunk["labevent_id"].isin(target_ids)
    ]

    if not matches.empty:
        results.append(matches)

if results:
    result = pd.concat(results, ignore_index=True)
    result.to_csv(OUTPUT, index=False)

    print(f"\nSaved original records to: {OUTPUT}")
    print("\nOriginal source values:")

    for item_id in TARGETS:
        subset = result[result["itemid"] == item_id]

        print(f"\nITEM ID: {item_id}")
        print(subset.head(8).to_string(index=False))
else:
    print("No matching records found.")
