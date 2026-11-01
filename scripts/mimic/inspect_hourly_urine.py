import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

ids = [226559, 226560]

chunks = pd.read_csv(
    path,
    compression="gzip",
    usecols=[
        "subject_id", "stay_id",
        "charttime", "itemid", "value"
    ],
    chunksize=1_000_000
)

parts = []

for chunk in chunks:
    chunk = chunk[chunk["itemid"].isin(ids)].copy()
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    chunk = chunk[chunk["value"] >= 0]

    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["hour"] = chunk["charttime"].dt.floor("h")

    parts.append(
        chunk.groupby(["stay_id", "hour", "itemid"])
        .agg(records=("value", "size"), volume_ml=("value", "sum"))
        .reset_index()
    )

hourly = pd.concat(parts, ignore_index=True)

hourly = hourly.groupby(["stay_id", "hour", "itemid"]).agg(
    records=("records", "sum"),
    volume_ml=("volume_ml", "sum")
).reset_index()

print("Hourly groups:", len(hourly))
print("\nHourly groups with multiple records:")
print(hourly[hourly["records"] > 1].head(20).to_string(index=False))

print("\nRecord count distribution:")
print(hourly["records"].value_counts().sort_index().head(15))

print("\nHourly volume summary:")
print(hourly["volume_ml"].describe())
