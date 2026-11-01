import pandas as pd

items = pd.read_csv(
    "data/real/raw/icu/d_items.csv.gz",
    compression="gzip"
)

ids = [226559, 226560, 226566, 226599, 226626, 226627, 227488, 227489]

result = items[items["itemid"].isin(ids)]

print(
    result[
        ["itemid", "label", "abbreviation", "linksto", "category", "unitname"]
    ].sort_values("itemid").to_string(index=False)
)
