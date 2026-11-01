import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"

ITEMS = [
    220045, 220210,
    223761, 223762,
    220050, 220051, 220052,
    220179, 220180, 220181,
    224688, 224689, 224690,
    229841
]

counts = {}

for chunk in pd.read_csv(
    PATH,
    compression="gzip",
    usecols=["itemid", "value", "valuenum", "valueuom"],
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk = chunk[chunk["itemid"].isin(ITEMS)]

    for itemid, group in chunk.groupby("itemid"):
        itemid = int(itemid)

        if itemid not in counts:
            counts[itemid] = {
                "records": 0,
                "numeric_values": 0,
                "units": {}
            }

        counts[itemid]["records"] += len(group)
        counts[itemid]["numeric_values"] += int(
            group["valuenum"].notna().sum()
        )

        for unit, count in group["valueuom"].fillna("NULL").value_counts().items():
            counts[itemid]["units"][str(unit)] = (
                counts[itemid]["units"].get(str(unit), 0) + int(count)
            )

print("\n========== VITAL-SIGN SOURCE VALIDATION ==========")

for itemid in ITEMS:
    result = counts.get(itemid)

    print(f"\nItem ID: {itemid}")

    if result is None:
        print("No records found")
        continue

    print("Records:", result["records"])
    print("Numeric values:", result["numeric_values"])
    print("Units:", result["units"])
