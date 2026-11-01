import pandas as pd

schedule_path = "data/real/processed/cv_assessment_schedule.csv"
vitals_path = "data/real/processed/cleaned_vitals.csv"
output_path = "data/real/processed/hourly_map.csv"

schedule = pd.read_csv(
    schedule_path,
    parse_dates=["assessment_time"]
)

vitals = pd.read_csv(
    vitals_path,
    usecols=["stay_id", "charttime", "feature", "value"],
    dtype={"stay_id": "int64"}
)

vitals = vitals[
    vitals["feature"].isin([
        "arterial_mean_bp",
        "nibp_mean_bp"
    ])
].copy()

vitals["charttime"] = pd.to_datetime(vitals["charttime"])
vitals["priority"] = vitals["feature"].map({
    "nibp_mean_bp": 1,
    "arterial_mean_bp": 2
})

# At identical timestamps, arterial MAP takes priority.
vitals = (
    vitals.sort_values(["charttime", "priority"])
    .drop_duplicates(["stay_id", "charttime"], keep="last")
)

vitals = vitals.rename(columns={
    "charttime": "map_charttime",
    "value": "map_mmhg",
    "feature": "map_source"
})

vitals = vitals[
    ["stay_id", "map_charttime", "map_mmhg", "map_source"]
].sort_values("map_charttime")

schedule = schedule.sort_values("assessment_time")

parts = []
chunk_size = 250_000

for start in range(0, len(schedule), chunk_size):
    chunk = schedule.iloc[start:start + chunk_size].copy()
    chunk = chunk.sort_values("assessment_time")

    merged = pd.merge_asof(
        chunk,
        vitals,
        left_on="assessment_time",
        right_on="map_charttime",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(minutes=60)
    )

    parts.append(merged)

    print(
        f"Processed {min(start + chunk_size, len(schedule)):,}"
        f" / {len(schedule):,} assessments"
    )

result = pd.concat(parts, ignore_index=True)

result["map_age_minutes"] = (
    result["assessment_time"] - result["map_charttime"]
).dt.total_seconds() / 60

result.to_csv(output_path, index=False)

print("\n========== MAP EXTRACTION ==========")
print("Assessments:", len(result))
print("With MAP:", result["map_mmhg"].notna().sum())
print("Missing MAP:", result["map_mmhg"].isna().sum())
print("\nMAP source:")
print(result["map_source"].value_counts(dropna=False).to_string())
print("\nMAP distribution:")
print(result["map_mmhg"].describe().round(2).to_string())
print("\nSaved:", output_path)
