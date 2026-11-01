import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

urine_ids = [226559, 226560, 226566, 226627, 226631, 227489]

chunks = pd.read_csv(
    path,
    compression="gzip",
    usecols=["itemid", "value", "valueuom"],
    chunksize=1_000_000
)

parts = []

for chunk in chunks:
    selected = chunk[chunk["itemid"].isin(urine_ids)]
    if not selected.empty:
        parts.append(selected)

data = pd.concat(parts, ignore_index=True)

print("RECORD COUNTS:")
print(data["itemid"].value_counts().sort_index())

print("\nUNIT DISTRIBUTION:")
print(data.groupby(["itemid", "valueuom"], dropna=False).size())

data["numeric_value"] = pd.to_numeric(data["value"], errors="coerce")

print("\nVALUE SUMMARY:")
print(
    data.groupby("itemid")["numeric_value"]
    .describe()
    .to_string()
)

print("\nNEGATIVE VALUES:")
print(data[data["numeric_value"] < 0].head(20).to_string(index=False))
