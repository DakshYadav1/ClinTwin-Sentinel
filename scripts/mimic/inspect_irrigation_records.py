import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

stays = [
    31127752, 32004192, 33010863, 33204925,
    33292306, 33676353, 34317198, 34364406,
    37782301, 37896702, 38986983
]

ids = [226559, 226560, 226566, 227488, 227489]

parts = []

for chunk in pd.read_csv(
    path,
    compression="gzip",
    usecols=[
        "subject_id", "stay_id", "charttime",
        "storetime", "itemid", "value", "valueuom"
    ],
    chunksize=1_000_000
):
    selected = chunk[
        chunk["stay_id"].isin(stays) &
        chunk["itemid"].isin(ids)
    ]

    if not selected.empty:
        parts.append(selected)

result = pd.concat(parts, ignore_index=True)

result = result.sort_values(["stay_id", "charttime", "itemid"])

print("MATCHING IRRIGATION AND URINE RECORDS:")
print(result.to_string(index=False))

print("\nCOUNTS BY ITEM:")
print(result.groupby("itemid").size())
