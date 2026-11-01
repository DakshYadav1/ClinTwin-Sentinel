import pandas as pd
from pathlib import Path

FILE = Path("data/real/processed/cleaned_labevents.csv")

EXPECTED_ROWS = 14_997_490
EXCLUDED_ID = 38928998

item_counts = {}
total_rows = 0
excluded_found = False

for chunk in pd.read_csv(
    FILE,
    usecols=["labevent_id", "itemid"],
    chunksize=500_000
):
    total_rows += len(chunk)

    excluded_found |= chunk["labevent_id"].eq(EXCLUDED_ID).any()

    counts = chunk["itemid"].value_counts()

    for item_id, count in counts.items():
        item_counts[int(item_id)] = (
            item_counts.get(int(item_id), 0) + int(count)
        )

print("CLEANED DATASET VALIDATION")
print("Total rows:", f"{total_rows:,}")
print("Expected rows:", f"{EXPECTED_ROWS:,}")
print("Excluded record present:", excluded_found)

print("\nLAB ITEM COUNTS:")
for item_id, count in sorted(item_counts.items()):
    print(f"{item_id}: {count:,}")

print("\nVALIDATION STATUS:")
print(
    "PASS" if total_rows == EXPECTED_ROWS and not excluded_found
    else "CHECK REQUIRED"
)
