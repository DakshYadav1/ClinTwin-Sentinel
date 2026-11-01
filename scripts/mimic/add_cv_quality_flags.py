import pandas as pd
import json

path = "data/real/processed/hourly_cardiovascular_sofa.csv"
report_path = "data/real/processed/cardiovascular_sofa_report.json"

df = pd.read_csv(path)

conditions = {
    "norepinephrine": df["active_norepinephrine"] > 5,
    "epinephrine": df["active_epinephrine"] > 5,
    "dopamine": df["active_dopamine"] > 40,
    "dobutamine": df["active_dobutamine"] > 40,
}

df["dose_review_flag"] = False

for medication, condition in conditions.items():
    df.loc[condition, "dose_review_flag"] = True

df.to_csv(path, index=False)

report = json.load(open(report_path))
report["dose_review_assessments"] = int(
    df["dose_review_flag"].sum()
)
report["dose_review_by_medication"] = {
    medication: int(condition.sum())
    for medication, condition in conditions.items()
}
report["extreme_dose_policy"] = (
    "Retained and flagged; no automatic exclusion."
)

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print("Dose-review assessments:", df["dose_review_flag"].sum())
print("\nReview flags by medication:")
for medication, condition in conditions.items():
    print(medication, int(condition.sum()))

print("\nExisting cardiovascular scores preserved.")
