import pandas as pd

FILE = "data/real/raw/icu/d_items.csv.gz"

items = pd.read_csv(FILE, compression="gzip")

keywords = [
    "heart rate",
    "respiratory rate",
    "temperature",
    "oxygen saturation",
    "spo2",
    "blood pressure",
    "mean blood pressure",
    "glasgow",
    "fio2",
    "fraction inspired oxygen",
    "urine output",
    "tidal volume",
    "ventilator"
]

mask = items["label"].str.contains(
    "|".join(keywords),
    case=False,
    na=False
)

result = items.loc[
    mask,
    ["itemid", "label", "abbreviation", "linksto", "category", "unitname"]
]

print("MATCHING ICU MEASUREMENT ITEMS")
print(result.sort_values(["category", "label"]).to_string(index=False))

print(f"\nTotal matching items: {len(result)}")
