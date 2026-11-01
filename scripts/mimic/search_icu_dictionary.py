import pandas as pd

FILE = "data/real/raw/icu/d_items.csv.gz"

items = pd.read_csv(FILE, compression="gzip")

patterns = [
    "saturation",
    "oxygen",
    "oximeter",
    "fraction inspired",
    "inspired oxygen",
    "oxygen delivery",
    "urine",
    "foley",
    "foley output"
]

for pattern in patterns:
    result = items[
        items["label"].str.contains(
            pattern, case=False, na=False
        )
    ]

    print(f"\n{'=' * 65}")
    print(f"SEARCH: {pattern}")
    print('=' * 65)

    print(
        result[
            ["itemid", "label", "linksto", "category", "unitname"]
        ].to_string(index=False)
    )
