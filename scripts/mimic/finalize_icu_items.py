import pandas as pd

items = pd.read_csv(
    "data/real/raw/icu/d_items.csv.gz",
    compression="gzip"
)

patterns = [
    "inspired",
    "fraction",
    "fio2",
    "foley",
    "void",
    "catheter",
    "urine"
]

for pattern in patterns:
    result = items[
        items["label"].str.contains(pattern, case=False, na=False)
    ]

    print(f"\n{'=' * 60}\n{pattern.upper()}\n{'=' * 60}")

    print(
        result[["itemid", "label", "linksto", "category", "unitname"]]
        .to_string(index=False)
    )
