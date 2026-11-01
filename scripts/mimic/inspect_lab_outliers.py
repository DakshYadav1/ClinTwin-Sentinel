import pandas as pd

FILE = "data/real/processed/filtered_labevents.csv"

LIMITS = {
    50813: 20,       # Lactate > 20 mmol/L
    50885: 50,       # Bilirubin > 50 mg/dL
    50912: 30,       # Creatinine > 30 mg/dL
    51265: 1000,     # Platelets > 1000 K/uL
    51301: 500       # WBC > 500 K/uL
}

extremes = {item: [] for item in LIMITS}
counts = {item: 0 for item in LIMITS}

for chunk in pd.read_csv(
    FILE,
    usecols=["itemid", "valuenum", "valueuom"],
    chunksize=500_000
):
    for item_id, limit in LIMITS.items():
        rows = chunk[
            (chunk["itemid"] == item_id) &
            (chunk["valuenum"] > limit)
        ]

        counts[item_id] += len(rows)

        if not rows.empty:
            extremes[item_id].extend(
                rows["valuenum"].nlargest(10).tolist()
            )

print("EXTREME VALUE COUNTS AND LARGEST OBSERVATIONS")

for item_id, limit in LIMITS.items():
    values = sorted(extremes[item_id], reverse=True)[:10]

    print(f"\nItem ID: {item_id} | Threshold: {limit}")
    print(f"Number of records above threshold: {counts[item_id]:,}")
    print("Largest values:", values)
