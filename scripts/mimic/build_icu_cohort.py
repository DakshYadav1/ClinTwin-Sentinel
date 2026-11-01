import pandas as pd

base = "data/real/raw"

patients = pd.read_csv(f"{base}/hosp/patients.csv.gz")
admissions = pd.read_csv(f"{base}/hosp/admissions.csv.gz")
icu = pd.read_csv(f"{base}/icu/icustays.csv.gz")

# Parse ICU timestamps
icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

# Calculate ICU duration from timestamps
icu["duration_hours"] = (
    icu["outtime"] - icu["intime"]
).dt.total_seconds() / 3600

# Join ICU stays to admission and patient information
cohort = icu.merge(
    admissions[["subject_id", "hadm_id", "admittime", "hospital_expire_flag"]],
    on=["subject_id", "hadm_id"],
    how="left",
    validate="many_to_one"
)

cohort = cohort.merge(
    patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
    on="subject_id",
    how="left",
    validate="many_to_one"
)

# Estimate age at ICU admission.
# MIMIC anchor_age values of 91 represent the 90+ age group.
year_difference = cohort["intime"].dt.year - cohort["anchor_year"]
cohort["age_at_icu"] = (
    cohort["anchor_age"] + year_difference
).clip(upper=91)

# Apply eligibility criteria
valid_time = (
    cohort["intime"].notna()
    & cohort["outtime"].notna()
    & (cohort["duration_hours"] > 0)
)

adult = cohort["age_at_icu"] >= 18
long_stay = cohort["duration_hours"] >= 24

eligible = valid_time & adult & long_stay

selected = cohort.loc[eligible].copy()

# Keep relevant fields for downstream processing
columns = [
    "subject_id", "hadm_id", "stay_id",
    "gender", "age_at_icu",
    "first_careunit", "last_careunit",
    "intime", "outtime", "duration_hours",
    "hospital_expire_flag"
]

selected[columns].to_csv(
    "data/real/processed/icu_cohort.csv",
    index=False
)

print("\n========== COHORT SUMMARY ==========")
print("Original ICU stays:", len(icu))
print("Missing/invalid ICU timestamps:", int((~valid_time).sum()))
print("Estimated age below 18:", int((cohort["age_at_icu"] < 18).sum()))
print("Adult stays:", int((valid_time & adult).sum()))
print("Adult stays >= 24 hours:", len(selected))
print("Unique patients:", selected["subject_id"].nunique())
print("Unique ICU stays:", selected["stay_id"].nunique())
print("\nCare unit distribution:")
print(selected["first_careunit"].value_counts().to_string())
print("\nHospital mortality flag:")
print(selected["hospital_expire_flag"].value_counts(dropna=False).to_string())
print("\nAge distribution:")
print(selected["age_at_icu"].describe().to_string())
print("\nDuration distribution (hours):")
print(selected["duration_hours"].describe().to_string())
print("\nSaved: data/real/processed/icu_cohort.csv")
print("====================================")
