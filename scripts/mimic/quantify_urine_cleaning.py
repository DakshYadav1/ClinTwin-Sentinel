import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"
ITEMS = [226559, 226560, 227488, 227489]

counts = {
    item: {
        "total": 0,
        "negative": 0,
        "above_2000": 0,
        "above_10000": 0,
        "above_50000": 0
    }
    for item in ITEMS
}

for chunk in pd.read_csv(
    PATH, compression="gzip",
    usecols=["stay_id", "charttime", "itemid", "value"],
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(ITEMS)]

    for itemid, group in chunk.groupby("itemid"):
        c = counts[int(itemid)]
        values = group["value"]

        c["total"] += len(group)
        c["negative"] += int((values < 0).sum())
        c["above_2000"] += int((values > 2000).sum())
        c["above_10000"] += int((values > 10000).sum())
        c["above_50000"] += int((values > 50000).sum())

print("\n========== CLEANING IMPACT ==========")
names = {
    226559: "Foley",
    226560: "Void",
    227488: "Irrigation input",
    227489: "Irrigation output"
}

for itemid, c in counts.items():
    print(f"\n{names[itemid]} ({itemid})")
    for key, value in c.items():
        print(f"{key}: {value}")
