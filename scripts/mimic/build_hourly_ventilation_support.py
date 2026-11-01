
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

PROCEDURE_FILE = (
    ROOT / "data/real/raw/mimiciv-3.1/icu/procedureevents.csv.gz"
)
SCHEDULE_FILE = ROOT / "data/real/processed/cv_assessment_schedule.csv"
OUTPUT_FILE = ROOT / "data/real/processed/hourly_ventilation_support.csv"
REPORT_FILE = ROOT / "data/real/processed/ventilation_support_report.json"

INVASIVE_ID = 225792
NONINVASIVE_ID = 225794
CHUNKSIZE = 250_000


def load_intervals():
    columns = [
        "stay_id", "itemid", "starttime", "endtime",
        "statusdescription"
    ]

    df = pd.read_csv(
        PROCEDURE_FILE,
        compression="gzip",
        usecols=columns
    )

    df = df[df["itemid"].isin([INVASIVE_ID, NONINVASIVE_ID])].copy()

    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")

    df = df.dropna(
        subset=["stay_id", "starttime", "endtime"]
    )

    df = df[df["endtime"] > df["starttime"]].copy()

    df["stay_id"] = df["stay_id"].astype("int64")

    return df


def build_interval_index(intervals):
    index = {}

    for (stay_id, itemid), group in intervals.groupby(
        ["stay_id", "itemid"], sort=False
    ):
        index[(int(stay_id), int(itemid))] = (
            group["starttime"].astype("int64").to_numpy(),
            group["endtime"].astype("int64").to_numpy(),
            group["statusdescription"].fillna("Unknown").to_numpy()
        )

    return index


def mark_support(times_ns, interval_data):
    starts, ends, statuses = interval_data

    # Merge intervals for efficient membership checks.
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]
    statuses = statuses[order]

    merged_starts = []
    merged_ends = []
    previous_end = None

    for start, end in zip(starts, ends):
        if previous_end is None or start > previous_end:
            merged_starts.append(start)
            merged_ends.append(end)
        else:
            merged_ends[-1] = max(merged_ends[-1], end)

        previous_end = max(previous_end, end) if previous_end is not None else end

    merged_starts = np.asarray(merged_starts, dtype=np.int64)
    merged_ends = np.asarray(merged_ends, dtype=np.int64)

    positions = np.searchsorted(merged_starts, times_ns, side="right") - 1

    active = np.zeros(len(times_ns), dtype=bool)
    valid = positions >= 0

    active[valid] = (
        times_ns[valid] < merged_ends[positions[valid]]
    )

    # Flag intervals whose source status was not FinishedRunning.
    review_starts = starts[np.isin(statuses, ["Paused", "Stopped", "Unknown"])]
    review_ends = ends[np.isin(statuses, ["Paused", "Stopped", "Unknown"])]

    review = np.zeros(len(times_ns), dtype=bool)

    if len(review_starts):
        review_order = np.argsort(review_starts)
        review_starts = review_starts[review_order]
        review_ends = review_ends[review_order]

        review_positions = (
            np.searchsorted(review_starts, times_ns, side="right") - 1
        )
        review_valid = review_positions >= 0
        review[review_valid] = (
            times_ns[review_valid] < review_ends[review_positions[review_valid]]
        )

    return active, review


def main():
    print("Loading ventilation intervals...")
    intervals = load_intervals()

    print("Building interval index...")
    interval_index = build_interval_index(intervals)

    print("Processing hourly assessment schedule...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    totals = {
        "assessment_rows": 0,
        "invasive_assessments": 0,
        "noninvasive_assessments": 0,
        "support_assessments": 0,
        "review_assessments": 0
    }

    for chunk in pd.read_csv(
        SCHEDULE_FILE,
        chunksize=CHUNKSIZE
    ):
        chunk["assessment_time"] = pd.to_datetime(
            chunk["assessment_time"]
        )

        chunk["stay_id"] = chunk["stay_id"].astype("int64")

        invasive = np.zeros(len(chunk), dtype=bool)
        noninvasive = np.zeros(len(chunk), dtype=bool)
        review = np.zeros(len(chunk), dtype=bool)

        for stay_id, positions in chunk.groupby("stay_id").indices.items():
            times = (
                chunk.iloc[positions]["assessment_time"]
                .astype("int64")
                .to_numpy()
            )

            inv_data = interval_index.get(
                (int(stay_id), INVASIVE_ID)
            )
            niv_data = interval_index.get(
                (int(stay_id), NONINVASIVE_ID)
            )

            if inv_data is not None:
                active, flagged = mark_support(times, inv_data)
                invasive[positions] = active
                review[positions] |= flagged

            if niv_data is not None:
                active, flagged = mark_support(times, niv_data)
                noninvasive[positions] = active
                review[positions] |= flagged

        chunk["invasive_ventilation"] = invasive.astype("int8")
        chunk["noninvasive_ventilation"] = noninvasive.astype("int8")
        chunk["ventilation_support"] = (
            invasive | noninvasive
        ).astype("int8")

        chunk["ventilation_overlap_flag"] = (
            invasive & noninvasive
        ).astype("int8")

        chunk["ventilation_status_review_flag"] = review.astype("int8")

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        totals["assessment_rows"] += len(chunk)
        totals["invasive_assessments"] += int(invasive.sum())
        totals["noninvasive_assessments"] += int(noninvasive.sum())
        totals["support_assessments"] += int(
            (invasive | noninvasive).sum()
        )
        totals["review_assessments"] += int(review.sum())

        print(
            f"Processed {totals['assessment_rows']:,} assessments..."
        )

    totals["source_intervals"] = len(intervals)
    totals["invasive_intervals"] = int(
        (intervals["itemid"] == INVASIVE_ID).sum()
    )
    totals["noninvasive_intervals"] = int(
        (intervals["itemid"] == NONINVASIVE_ID).sum()
    )

    REPORT_FILE.write_text(
        pd.Series(totals).to_json(indent=4),
        encoding="utf-8"
    )

    print("\nVentilation processing complete.")
    print(pd.Series(totals).to_string())
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
