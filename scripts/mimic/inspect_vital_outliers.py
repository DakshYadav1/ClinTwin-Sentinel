import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"

BOUNDS = {
    220045: (20, 250),
    220210: (1, 100),
    223761: (77, 113),
    223762: (25, 45),
    220050: (40, 300),
    220051: (10, 200),
    220052: (20, 250),
    220179: (40, 300),
    220180: (10, 200),
    220181: (20, 250),
    224688: (1, 100),
    224689: (1, 100),
    224690: (1, 100),
    229841: (21, 100)
}

columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "storetime", "itemid",
    "value", "valuenum", "valueuom"
]

counts = {item: {"below": 0, "above": 0, "zero": 0} for item in BOUNDS}
candidates = []

for chunk in pd.read_csv(
    PATH, compression="gzip",
    usecols=columns, chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(BOUNDS)]

    for itemid, group in chunk.groupby("itemid"):
        itemid = int(itemid)
        low, high = BOUNDS[itemid]
        values = group["valuenum"]

        counts[itemid]["below"] += int((values < low).sum())
        counts[itemid]["above"] += int((values > high).sum())

        if itemid == 224689:
            counts[itemid]["zero"] += int((values == 0).sum())

        suspicious = group[(values < low) | (values > high)].copy()

        if itemid == 224689:
            suspicious = pd.concat([
                suspicious,
                group[values == 0]
            ]).drop_duplicates()

        if not suspicious.empty:
            candidates.append(suspicious)

print("\n========== OUT-OF-RANGE COUNTS ==========")
for itemid, result in counts.items():
    print(itemid, result)

if candidates:
    df = pd.concat(candidates, ignore_index=True)

    print("\n========== MOST EXTREME RECORDS ==========")

    for itemid, group in df.groupby("itemid"):
        low, high = BOUNDS[int(itemid)]

        print(f"\nITEM {int(itemid)} | Screening range: {low}–{high}")

        print("\nLowest observations:")
        print(
            group.sort_values("valuenum").head(5)[columns]
            .to_string(index=False)
        )

        print("\nHighest observations:")
        print(
            group.sort_values("valuenum", ascending=False).head(5)[columns]
            .to_string(index=False)
        )

        if int(itemid) == 224689:
            print("\nZero respiratory-rate examples:")
            print(
                group[group["valuenum"] == 0].head(5)[columns]
                .to_string(index=False)
            )
