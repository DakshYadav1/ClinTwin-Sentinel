import pandas as pd

cohort_path = "data/real/processed/strict_icu_cohort.csv"
output_path = "data/real/processed/cv_assessment_schedule.csv"

cohort = pd.read_csv(cohort_path, parse_dates=["intime", "outtime"])

rows = []

for r in cohort.itertuples(index=False):
    start = r.intime + pd.Timedelta(hours=1)

    if pd.isna(r.outtime) or start > r.outtime:
        continue

    times = pd.date_range(
        start=start,
        end=r.outtime,
        freq="h"
    )

    for t in times:
        rows.append({
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
            "stay_id": r.stay_id,
            "assessment_time": t
        })

schedule = pd.DataFrame(rows)
schedule.to_csv(output_path, index=False)

print("========== ASSESSMENT SCHEDULE ==========")
print("ICU stays:", cohort["stay_id"].nunique())
print("Stays with assessments:", schedule["stay_id"].nunique())
print("Hourly assessments:", len(schedule))
print("Unique patients:", schedule["subject_id"].nunique())
print("First assessment:", schedule["assessment_time"].min())
print("Last assessment:", schedule["assessment_time"].max())
print("Duplicate stay-time pairs:",
      schedule.duplicated(["stay_id", "assessment_time"]).sum())
print("Saved:", output_path)
