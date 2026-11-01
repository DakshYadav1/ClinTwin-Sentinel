import pandas as pd

path = "data/real/raw/icu/d_items.csv.gz"
items = pd.read_csv(path)

terms = [
    "ventilator", "ventilation", "vent mode",
    "vent type", "mechanical ventilation",
    "invasive ventilation", "extubation",
    "intubation", "airway type", "oxygen delivery"
]

mask = items["label"].fillna("").str.contains(
    "|".join(terms), case=False, regex=True
)

cols = [
    c for c in [
        "itemid", "label", "category",
        "unitname", "param_type"
    ] if c in items.columns
]

print("========== VENTILATION DICTIONARY ==========")
print(items.loc[mask, cols].to_string(index=False))
print("\nMatching dictionary items:", int(mask.sum()))
