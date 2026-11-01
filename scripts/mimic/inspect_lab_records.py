import pandas as pd

FILE = "data/real/processed/filtered_labevents.csv"

TARGETS = {
    50813: 20,
    50885: 50,
    50912: 30,
    51265: 1000,
    51301: 500
}

records = []

for chunk in pd.read_csv(
    FILE,
    usecols=[
        "subject_id", "hadm_id", "itemid", "charttime",
        "valuenum", "valueuom", "ref_range_lower",
        "ref_range_upper", "flag"
    ],
    chunksize=500_000
):
    for item_id, threshold in TARGETS.items():
        matches = chunk[
            (chunk["itemid"] == item_id) &
            (chunk["valuenum"] > threshold)
        ]

        if not matches.empty:
            records.append(matches)

if records:
    result = pd.concat(records, ignore_index=True)

    result = result.sort_values(
        ["itemid", "valuenum"],
        ascending=[True, False]
    )

    output = "data/real/processed/extreme_lab_records.csv"
    result.to_csv(output, index=False)

    print("Extreme records saved:", output)

    for item_id in TARGETS:
        subset = result[result["itemid"] == item_id]

        print(f"\nITEM ID: {item_id}")
        print(
            subset.head(5).to_string(index=False)
            if not subset.empty else "No matching records"
        )
else:
    print("No extreme records found.")
