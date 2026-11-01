from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/real/processed/filtered_labevents.csv")
CHUNK_SIZE = 500_000

stats = {}

print("Inspecting laboratory value ranges...")

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["itemid", "valuenum"],
    chunksize=CHUNK_SIZE
):
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    for item_id, group in chunk.groupby("itemid"):
        item_id = int(item_id)

        if item_id not in stats:
            stats[item_id] = {
                "min": float("inf"),
                "max": float("-inf"),
                "count": 0,
                "missing": 0,
                "sample": []
            }

        valid = group["valuenum"].dropna()

        if not valid.empty:
            stats[item_id]["min"] = min(
                stats[item_id]["min"], valid.min()
            )
            stats[item_id]["max"] = max(
                stats[item_id]["max"], valid.max()
            )
            stats[item_id]["count"] += len(valid)

            sample = valid.sample(
                n=min(1000, len(valid)),
                random_state=42
            )
            stats[item_id]["sample"].extend(sample.tolist())

        stats[item_id]["missing"] += int(
            group["valuenum"].isna().sum()
        )

print("\nITEM ID | MIN | MEDIAN (SAMPLE) | MAX | VALID | MISSING")

for item_id, result in sorted(stats.items()):
    median = pd.Series(result["sample"]).median()

    print(
        f"{item_id} | "
        f"{result['min']:.3f} | "
        f"{median:.3f} | "
        f"{result['max']:.3f} | "
        f"{result['count']:,} | "
        f"{result['missing']:,}"
    )

print("\nValidation completed.")
