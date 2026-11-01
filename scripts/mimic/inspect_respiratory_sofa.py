import pandas as pd

icu = pd.read_csv("data/real/raw/icu/d_items.csv.gz")
labs = pd.read_csv("data/real/raw/hosp/d_labitems.csv.gz")

terms = [
    "inspired",
    "fraction of inspired",
    "oxygen device",
    "oxygen flow",
    "ventilator",
    "pao2",
]

print("\n========== RESPIRATORY ICU ITEMS ==========")

for term in terms:
    matches = icu[
        icu["label"].str.contains(term, case=False, na=False)
    ]

    if not matches.empty:
        print(f"\n--- {term.upper()} ---")
        cols = ["itemid", "label", "category", "unitname", "linksto"]
        print(matches[[c for c in cols if c in matches.columns]]
              .to_string(index=False))

print("\n========== PAO2 LAB ITEM ==========")

print(
    labs[labs["itemid"] == 50821]
    [["itemid", "label", "fluid", "category"]]
    .to_string(index=False)
)

print("\nInspection complete.")
