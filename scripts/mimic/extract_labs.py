
from pathlib import Path
import pandas as pd
import json

# Project paths
INPUT_FILE = Path("data/real/raw/mimiciv-3.1/hosp/labevents.csv.gz")
OUTPUT_DIR = Path("data/real/processed")
OUTPUT_FILE = OUTPUT_DIR / "filtered_labevents.csv"

# Candidate laboratory item IDs
LAB_ITEM_IDS = [
    50813, 52442, 53154,       # Lactate
    51300, 51301, 51755, 51756, # WBC
    50912, 52546,              # Creatinine
    50885, 53089,              # Total bilirubin
    51265, 53189               # Platelets
]

COLUMNS = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "itemid",
    "charttime",
    "valuenum",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag"
]

CHUNK_SIZE = 500_000


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally appending to an old extraction
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. Review it before rerunning."
        )

    counts = {str(item): 0 for item in LAB_ITEM_IDS}
    total_rows = 0
    matched_rows = 0
    first_chunk = True

    print("Starting laboratory extraction...")
    print(f"Input: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                INPUT_FILE,
                usecols=COLUMNS,
                chunksize=CHUNK_SIZE,
                compression="gzip",
                low_memory=True
            ),
            start=1
        ):
            total_rows += len(chunk)

            filtered = chunk[
                chunk["itemid"].isin(LAB_ITEM_IDS)
            ].copy()

            if not filtered.empty:
                filtered.to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False
                )
                first_chunk = False
                matched_rows += len(filtered)

                for item_id, count in filtered["itemid"].value_counts().items():
                    counts[str(item_id)] += int(count)

            print(
                f"Chunk {chunk_number}: "
                f"{total_rows:,} rows scanned | "
                f"{matched_rows:,} matches"
            )

        if first_chunk:
            print("WARNING: No matching laboratory records found.")
        else:
            print(f"\nFiltered data saved: {OUTPUT_FILE}")

        report = {
            "source": str(INPUT_FILE),
            "total_rows_scanned": total_rows,
            "matched_rows": matched_rows,
            "item_id_counts": counts
        }

        report_file = OUTPUT_DIR / "lab_extraction_report.json"
        report_file.write_text(json.dumps(report, indent=2))

        print(f"Report saved: {report_file}")
        print("\nItem ID counts:")
        for item_id, count in counts.items():
            print(f"{item_id}: {count:,}")

    except Exception:
        # Do not leave an apparently complete output after a failed run
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        raise


if __name__ == "__main__":
    main()
