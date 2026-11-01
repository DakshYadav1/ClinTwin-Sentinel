
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/real/raw/mimiciv-3.1/icu"
PROCESSED = ROOT / "data/real/processed"

PAIRED_FILE = PROCESSED / "paired_pao2_fio2.csv"
SCHEDULE_FILE = PROCESSED / "cv_assessment_schedule.csv"
PROCEDURE_FILE = RAW / "procedureevents.csv.gz"

OBSERVATION_FILE = PROCESSED / "respiratory_scored_observations.csv"
OUTPUT_FILE = PROCESSED / "hourly_respiratory_sofa.csv"
REPORT_FILE = PROCESSED / "respiratory_sofa_report.json"

INVASIVE_ID = 225792
NONINVASIVE_ID = 225794
CARRY_FORWARD_HOURS = 6


def respiratory_score(pf_ratio, support):
    if pd.isna(pf_ratio):
        return np.nan

    if pf_ratio >= 400:
        return 0
    if pf_ratio >= 300:
        return 1
    if pf_ratio >= 200:
        return 2
    if support:
        if pf_ratio >= 100:
            return 3
        return 4

    return 2


def load_support_intervals():
    columns = [
        "stay_id", "itemid", "starttime", "endtime",
        "statusdescription"
    ]

    df = pd.read_csv(
        PROCEDURE_FILE,
        compression="gzip",
        usecols=columns
    )

    df = df[df["itemid"].isin(
        [INVASIVE_ID, NONINVASIVE_ID]
    )].copy()

    df["starttime"] = pd.to_datetime(
        df["starttime"], errors="coerce"
    )
    df["endtime"] = pd.to_datetime(
        df["endtime"], errors="coerce"
    )

    df = df.dropna(
        subset=["stay_id", "starttime", "endtime"]
    )

    df = df[df["endtime"] > df["starttime"]]

    return df


def support_at_observation(observations, intervals):
    observations = observations.copy()

    observations["invasive_ventilation"] = 0
    observations["noninvasive_ventilation"] = 0
    observations["ventilation_overlap_flag"] = 0
    observations["ventilation_status_review_flag"] = 0

    for itemid, feature in [
        (INVASIVE_ID, "invasive_ventilation"),
        (NONINVASIVE_ID, "noninvasive_ventilation")
    ]:
        source = intervals[intervals["itemid"] == itemid]

        for stay_id, group in source.groupby("stay_id"):
            positions = observations.index[
                observations["stay_id"] == stay_id
            ]

            if len(positions) == 0:
                continue

            starts = np.sort(
                group["starttime"].astype("int64").to_numpy()
            )
            ends = group["endtime"].astype("int64").to_numpy()

            order = np.argsort(
                group["starttime"].astype("int64").to_numpy()
            )
            ends = ends[order]

            # Merge overlapping intervals.
            merged_starts = []
            merged_ends = []

            for start, end in zip(starts, ends):
                if not merged_ends or start > merged_ends[-1]:
                    merged_starts.append(start)
                    merged_ends.append(end)
                else:
                    merged_ends[-1] = max(
                        merged_ends[-1], end
                    )

            times = (
                observations.loc[positions, "charttime"]
                .astype("int64").to_numpy()
            )

            idx = (
                np.searchsorted(
                    merged_starts, times, side="right"
                ) - 1
            )

            active = np.zeros(len(times), dtype=bool)
            valid = idx >= 0

            active[valid] = (
                times[valid] < np.asarray(merged_ends)[idx[valid]]
            )

            observations.loc[
                positions, feature
            ] = active.astype("int8")

    observations["ventilation_support"] = (
        observations["invasive_ventilation"].astype(bool) |
        observations["noninvasive_ventilation"].astype(bool)
    ).astype("int8")

    observations["ventilation_overlap_flag"] = (
        observations["invasive_ventilation"].astype(bool) &
        observations["noninvasive_ventilation"].astype(bool)
    ).astype("int8")

    return observations


def build_observation_scores():
    print("Loading paired PaO2-FiO2 observations...")

    observations = pd.read_csv(
    PAIRED_FILE,
    parse_dates=["charttime"]
)

    observations = observations[
        observations["fio2_fraction"].notna()
    ].copy()

    observations = observations[
        observations["fio2_fraction"].between(0.20, 1.0)
    ].copy()

    observations["pf_ratio"] = (
        observations["pao2_mmhg"] /
        observations["fio2_fraction"]
    )

    print("Loading ventilation intervals...")
    intervals = load_support_intervals()

    print("Checking support at PaO2 measurement times...")
    observations = support_at_observation(
        observations, intervals
    )

    observations["respiratory_sofa"] = observations.apply(
        lambda row: respiratory_score(
            row["pf_ratio"],
            bool(row["ventilation_support"])
        ),
        axis=1
    )

    observations["pf_extreme_flag"] = (
        observations["pf_ratio"] > 600
    ).astype("int8")

    observations["respiratory_sofa"] = (
        observations["respiratory_sofa"].astype("int8")
    )

    observations.to_csv(
        OBSERVATION_FILE, index=False
    )

    print(
        "Scored observations:",
        len(observations)
    )

    return observations


def build_hourly_scores(observations):
    print("Loading hourly assessment schedule...")

    schedule = pd.read_csv(
        SCHEDULE_FILE,
        parse_dates=["assessment_time"]
    )

    schedule = schedule.sort_values(
        ["assessment_time", "stay_id"]
    )

    observations = observations.rename(
        columns={"charttime": "pao2_time"}
    )

    observations = observations.sort_values(
        ["pao2_time", "stay_id"]
    )

    print("Matching latest respiratory score to each hour...")

    hourly = pd.merge_asof(
        schedule,
        observations[[
            "stay_id",
            "pao2_time",
            "respiratory_sofa",
            "pf_ratio",
            "invasive_ventilation",
            "noninvasive_ventilation",
            "ventilation_overlap_flag",
            "pf_extreme_flag"
        ]],
        left_on="assessment_time",
        right_on="pao2_time",
        by="stay_id",
        direction="backward",
        tolerance=pd.Timedelta(
            hours=CARRY_FORWARD_HOURS
        )
    )

    hourly["respiratory_observation_age_minutes"] = (
        (
            hourly["assessment_time"] -
            hourly["pao2_time"]
        ).dt.total_seconds() / 60
    )

    hourly["respiratory_observation_missing"] = (
        hourly["respiratory_sofa"].isna()
    ).astype("int8")

    hourly.to_csv(
        OUTPUT_FILE, index=False
    )

    return hourly


def main():
    observations = build_observation_scores()
    hourly = build_hourly_scores(observations)

    report = {
        "scored_observations": int(len(observations)),
        "hourly_assessments": int(len(hourly)),
        "hourly_scores_available": int(
            hourly["respiratory_sofa"].notna().sum()
        ),
        "hourly_scores_missing": int(
            hourly["respiratory_sofa"].isna().sum()
        ),
        "hourly_score_distribution": {
            str(int(k)): int(v)
            for k, v in hourly[
                "respiratory_sofa"
            ].value_counts().sort_index().items()
        },
        "extreme_pf_observations": int(
            observations["pf_extreme_flag"].sum()
        ),
        "ventilation_overlap_observations": int(
            observations["ventilation_overlap_flag"].sum()
        ),
        "pairing_window_minutes": 60,
        "hourly_carry_forward_hours": CARRY_FORWARD_HOURS
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nRespiratory SOFA processing complete.")
    print(json.dumps(report, indent=4))
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
