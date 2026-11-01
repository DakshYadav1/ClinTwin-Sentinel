import pandas as pd
import os

path = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"

item_ids = [225792, 225794, 223836, 223848, 223849]

columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "itemid", "value", "valuenum"
]

parts = []

print("Scanning chartevents for ventilation indicators...")

for chunk in pd.read_csv(path, usecols=columns, chunksize=1_000_000):
    selected = chunk[chunk["itemid"].isin(item_ids)]
    if not selected.empty:
        parts.append(selected)

df = pd.concat(parts, ignore_index=True)

print("\n========== VENTILATION RECORDS ==========")
print("Total records:", len(df))

for item in item_ids:
    subset = df[df["itemid"] == item]
    print(f"\nItem ID: {item}")
    print("Records:", len(subset))
    print("Unique stays:", subset["stay_id"].nunique())
    print("Distinct values:")
    print(subset["value"].value_counts(dropna=False).head(20).to_string())
    print("\nExamples:")
    print(subset.head(5).to_string(index=False))

print("\nProcedure events file exists:",
      os.path.exists("data/real/raw/mimiciv-3.1/icu/procedureevents.csv.gz"))
