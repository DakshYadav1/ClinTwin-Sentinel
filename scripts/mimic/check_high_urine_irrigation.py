import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"
ITEMS = [226559, 226560, 227488, 227489]

parts = []

for chunk in pd.read_csv(
    PATH, compression="gzip",
    usecols=["stay_id", "charttime", "itemid", "value"],
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    chunk["charttime"] = pd.to_datetime(chunk["charttime"])

    chunk = chunk[chunk["itemid"].isin(ITEMS)].copy()
    chunk["hour"] = chunk["charttime"].dt.floor("h")
    parts.append(chunk)

df = pd.concat(parts, ignore_index=True)

df["is_irrigation"] = df["itemid"].isin([227488, 227489])

irrigation_hours = set(
    map(tuple, df.loc[
        df["is_irrigation"], ["stay_id", "hour"]
    ].drop_duplicates().to_numpy())
)

high = df[
    df["itemid"].isin([226559, 226560]) &
    (df["value"] > 2000)
].copy()

high["same_hour_irrigation"] = [
    (stay, hour) in irrigation_hours
    for stay, hour in zip(high["stay_id"], high["hour"])
]

for itemid, name in [(226559, "Foley"), (226560, "Void")]:
    group = high[high["itemid"] == itemid]

    print(f"\n{name}:")
    print("High records:", len(group))
    print("Same-hour irrigation:", int(group["same_hour_irrigation"].sum()))
    print("No same-hour irrigation:", int((~group["same_hour_irrigation"]).sum()))

print("\nHigh records with same-hour irrigation:")
print(
    high[high["same_hour_irrigation"]][
        ["stay_id", "charttime", "itemid", "value"]
    ].sort_values(["stay_id", "charttime"]).to_string(index=False)
)
