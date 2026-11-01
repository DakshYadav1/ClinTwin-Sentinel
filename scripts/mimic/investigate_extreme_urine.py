import pandas as pd

processed = pd.read_csv(
    "data/real/processed/hourly_urine_output.csv"
)

extreme = processed[
    processed["total_recorded_urine_ml"] > 5000
].copy()

extreme["hour"] = pd.to_datetime(extreme["hour"])

print("EXTREME HOURLY GROUPS:")
print(extreme.to_string(index=False))

keys = set(
    zip(extreme["stay_id"], extreme["hour"])
)

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

parts = []

for chunk in pd.read_csv(
    path,
    compression="gzip",
    usecols=[
        "subject_id", "hadm_id", "stay_id",
        "charttime", "storetime", "itemid",
        "value", "valueuom"
    ],
    chunksize=1_000_000
):
    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["hour"] = chunk["charttime"].dt.floor("h")

    mask = [
        (stay, hour) in keys
        for stay, hour in zip(chunk["stay_id"], chunk["hour"])
    ]

    selected = chunk[mask]

    if not selected.empty:
        parts.append(selected)

original = pd.concat(parts, ignore_index=True)

print("\nORIGINAL SOURCE RECORDS:")
print(
    original.sort_values(["stay_id", "charttime"])
    .to_string(index=False)
)
