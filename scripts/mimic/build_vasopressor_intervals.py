import pandas as pd

source = "data/real/processed/cleaned_vasopressors.csv"
output = "data/real/processed/vasopressor_intervals.csv"

df = pd.read_csv(source)

df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")

df["duration_minutes"] = (
    df["endtime"] - df["starttime"]
).dt.total_seconds() / 60

before = len(df)

# Boluses are not continuous infusion intervals.
df = df[df["statusdescription"] != "Bolus"].copy()

# Require a valid, positive-duration interval.
df = df[
    df["starttime"].notna()
    & df["endtime"].notna()
    & (df["duration_minutes"] > 0)
].copy()

df.to_csv(output, index=False)

print("========== INTERVAL DATASET ==========")
print("Original records:", before)
print("Continuous infusion intervals:", len(df))
print("Excluded:", before - len(df))
print("Invalid durations:", (df["duration_minutes"] <= 0).sum())
print("\nMedication counts:")
print(df["medication"].value_counts().to_string())
print("\nDuration (minutes):")
print(df["duration_minutes"].describe().round(2).to_string())
print("\nSaved:", output)
