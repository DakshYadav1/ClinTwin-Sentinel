import pandas as pd

FILE = "data/real/processed/suspicious_lab_source_records.csv"

df = pd.read_csv(FILE)

for item_id in [50813, 50885, 50912, 51265, 51301]:
    subset = df[df["itemid"] == item_id].copy()

    subset = subset.sort_values(
        "valuenum",
        ascending=False
    ).head(20)

    print(f"\n{'=' * 75}")
    print(f"ITEM ID: {item_id}")
    print(f"{'=' * 75}")

    columns = [
        "labevent_id",
        "value",
        "valuenum",
        "valueuom",
        "ref_range_lower",
        "ref_range_upper",
        "flag",
        "comments"
    ]

    print(subset[columns].to_string(index=False))
