import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

ids = [226559, 226560]

chunks = pd.read_csv(
    path,
    compression="gzip",
    usecols=[
        "subject_id", "hadm_id", "stay_id",
        "charttime", "storetime",
        "itemid", "value", "valueuom"
    ],
    chunksize=1_000_000
)

negative = []

for chunk in chunks:
    chunk["numeric_value"] = pd.to_numeric(
        chunk["value"], errors="coerce"
    )

    selected = chunk[
        chunk["itemid"].isin(ids) &
        (chunk["numeric_value"] < 0)
    ]

    if not selected.empty:
        negative.append(selected)

result = pd.concat(negative, ignore_index=True)

print("Total negative records:", len(result))
print("\nNEGATIVE VALUE FREQUENCIES:")
print(result.groupby(["itemid", "numeric_value"]).size()
      .sort_values(ascending=False).head(25))

print("\nSAMPLE ORIGINAL RECORDS:")
print(result.head(30).to_string(index=False))
