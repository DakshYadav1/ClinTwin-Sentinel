import pandas as pd

path = "data/real/processed/cleaned_vitals.csv"
chunksize = 1_000_000

expected = {
    "heart_rate": (20, 250),
    "respiratory_rate": (1, 100),
    "temperature": ((77 - 32) * 5 / 9, (113 - 32) * 5 / 9),
    "arterial_systolic_bp": (40, 300),
    "arterial_diastolic_bp": (10, 200),
    "arterial_mean_bp": (20, 250),
    "nibp_systolic_bp": (40, 300),
    "nibp_diastolic_bp": (10, 200),
    "nibp_mean_bp": (20, 250),
}

counts = {}
missing = {}
minimum = {}
maximum = {}
invalid_units = 0
duplicates = 0
total = 0

for chunk in pd.read_csv(path, chunksize=chunksize):
    total += len(chunk)

    for col in ["subject_id", "stay_id", "charttime", "itemid", "feature", "value", "unit"]:
        missing[col] = missing.get(col, 0) + chunk[col].isna().sum()

    for feature, group in chunk.groupby("feature"):
        counts[feature] = counts.get(feature, 0) + len(group)
        minimum[feature] = min(minimum.get(feature, float("inf")), group["value"].min())
        maximum[feature] = max(maximum.get(feature, float("-inf")), group["value"].max())

        if feature in expected:
            low, high = expected[feature]
            if not group["value"].between(low, high).all():
                print(f"OUT-OF-RANGE VALUES FOUND: {feature}")

    invalid_units += (
        chunk.loc[chunk["feature"] == "temperature", "unit"] != "\u00b0C"
    ).sum()

    duplicates += chunk.duplicated(
        subset=["stay_id", "charttime", "itemid", "value"]
    ).sum()

print("\n========== VITAL VALIDATION ==========")
print("Total rows:", total)
print("Rows by feature:", counts)
print("Missing values:", missing)
print("Minimum values:", minimum)
print("Maximum values:", maximum)
print("Temperature rows not in \u00b0C:", invalid_units)
print("Duplicate records:", duplicates)
print("=====================================")

