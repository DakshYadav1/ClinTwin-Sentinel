import pandas as pd

files = {
    "patients": "data/real/raw/hosp/patients.csv.gz",
    "admissions": "data/real/raw/hosp/admissions.csv.gz",
    "icustays": "data/real/raw/icu/icustays.csv.gz",
}

for name, path in files.items():
    df = pd.read_csv(path)

    print(f"\n========== {name.upper()} ==========")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    if name == "patients":
        print("\nUnique patients:", df["subject_id"].nunique())
        print("Gender distribution:")
        print(df["gender"].value_counts(dropna=False).to_string())

    elif name == "admissions":
        print("\nUnique patients:", df["subject_id"].nunique())
        print("Admission time range:",
              df["admittime"].min(), "to", df["admittime"].max())

    elif name == "icustays":
        intime = pd.to_datetime(df["intime"])
        outtime = pd.to_datetime(df["outtime"])
        duration = (outtime - intime).dt.total_seconds() / 3600

        print("\nUnique patients:", df["subject_id"].nunique())
        print("Unique ICU stays:", df["stay_id"].nunique())
        print("ICU time range:", intime.min(), "to", outtime.max())
        print("Stay duration (hours):")
        print(duration.describe().to_string())
        print("Stays >= 24 hours:", (duration >= 24).sum())
