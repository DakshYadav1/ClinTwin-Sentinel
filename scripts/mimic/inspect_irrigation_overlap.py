import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

ids = [226559, 226560, 226566, 227488, 227489]

chunks = pd.read_csv(
    path,
    compression="gzip",
    usecols=["stay_id", "charttime", "itemid", "value"],
    chunksize=1_000_000
)

parts = []

for chunk in chunks:
    chunk = chunk[chunk["itemid"].isin(ids)].copy()

    chunk["charttime"] = pd.to_datetime(chunk["charttime"])
    chunk["hour"] = chunk["charttime"].dt.floor("h")

    grouped = chunk.groupby(
        ["stay_id", "hour", "itemid"]
    ).agg(
        records=("value", "size"),
        volume_ml=("value", "sum")
    ).reset_index()

    parts.append(grouped)

data = pd.concat(parts, ignore_index=True)

hourly = data.pivot_table(
    index=["stay_id", "hour"],
    columns="itemid",
    values="records",
    aggfunc="sum",
    fill_value=0
).reset_index()

for item in ids:
    if item not in hourly.columns:
        hourly[item] = 0

print("TOTAL STAY-HOUR GROUPS:", len(hourly))

print("\nOVERLAP COUNTS:")
print("Foley + irrigation input:",
      ((hourly[226559] > 0) & (hourly[227488] > 0)).sum())

print("Foley + irrigation output:",
      ((hourly[226559] > 0) & (hourly[227489] > 0)).sum())

print("Irrigation input + output:",
      ((hourly[227488] > 0) & (hourly[227489] > 0)).sum())

print("Combined urine/irrigant output:",
      (hourly[226566] > 0).sum())

print("\nIRRIGATION INPUT/OUTPUT RECORD COUNTS:")
print(data.groupby("itemid")["records"].sum())
