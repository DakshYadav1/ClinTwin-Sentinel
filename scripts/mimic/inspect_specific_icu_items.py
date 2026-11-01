import pandas as pd

FILE = "data/real/raw/icu/d_items.csv.gz"

items = pd.read_csv(FILE, compression="gzip")

keywords = [
    "oxygen saturation",
    "pulse oximetry",
    "spo2",
    "urine output",
    "urine volume",
    "fio2"
]

for keyword in keywords:
    print(f"\n{'=' * 60}")
    print("SEARCH:", keyword)
    print('=' * 60)

    result = items[
        items["label"].str.contains(
            keyword, case=False, na=False
        )
    ]

    print(
        result[
            ["itemid", "label", "abbreviation",
             "linksto", "category", "unitname"]
        ].to_string(index=False)
    )
