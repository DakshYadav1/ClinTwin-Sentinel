from pathlib import Path
import pandas as pd
import json

INPUT = Path("data/real/processed/filtered_labevents.csv")
OUTPUT = Path("data/real/processed/cleaned_labevents.csv")
REPORT = Path("data/real/processed/lab_cleaning_report.json")

INVALID_LABEVENT_IDS = {38928998}

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Review before rerunning."
    )

total_rows = 0
excluded_rows = 0
missing_values = 0
first_chunk = True

for chunk in pd.read_csv(INPUT, chunksize=500_000):

    total_rows += len(chunk)

    invalid = chunk["labevent_id"].isin(INVALID_LABEVENT_IDS)

    excluded_rows += int(invalid.sum())

    cleaned = chunk.loc[~invalid].copy()

    missing_values += int(cleaned["valuenum"].isna().sum())

    cleaned.to_csv(
        OUTPUT,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

report = {
    "source": str(INPUT),
    "output": str(OUTPUT),
    "total_input_rows": total_rows,
    "excluded_rows": excluded_rows,
    "retained_rows": total_rows - excluded_rows,
    "retained_missing_valuenum": missing_values,
    "excluded_labevent_ids": sorted(INVALID_LABEVENT_IDS)
}

REPORT.write_text(json.dumps(report, indent=2))

print("\nCleaning completed.")
print(json.dumps(report, indent=2))
