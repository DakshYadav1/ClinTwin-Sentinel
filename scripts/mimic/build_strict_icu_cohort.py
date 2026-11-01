import pandas as pd

source = "data/real/processed/icu_cohort.csv"
output = "data/real/processed/strict_icu_cohort.csv"

df = pd.read_csv(source)

allowed_units = [
    "Medical Intensive Care Unit (MICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Coronary Care Unit (CCU)",
    "Trauma SICU (TSICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Intensive Care Unit (ICU)",
]

strict = df[df["first_careunit"].isin(allowed_units)].copy()
strict.to_csv(output, index=False)

print("\n========== STRICT ICU COHORT ==========")
print("Original eligible stays:", len(df))
print("Strict ICU stays:", len(strict))
print("Excluded stays:", len(df) - len(strict))
print("Unique patients:", strict["subject_id"].nunique())
print("\nIncluded units:")
print(strict["first_careunit"].value_counts().to_string())
print("\nHospital mortality:")
print(strict["hospital_expire_flag"].value_counts().to_string())
print("\nSaved:", output)
print("======================================")
