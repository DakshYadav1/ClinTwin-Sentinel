"""
Build medication-specific vasopressor features for strict MIMIC deterioration anchors.

Inputs:
  data/real/processed/mimic_features_strict.csv
  data/real/processed/vasopressor_intervals.csv

Output:
  data/real/processed/mimic_features_with_vasopressors.csv
  data/real/processed/vasopressor_feature_report.json

Operational definition:
  - Eligible statuses: ChangeDose/Rate and FinishedRunning.
  - Active when starttime <= assessment_time < endtime.
  - Rates are retained in mcg/kg/min and are NOT pooled across drugs.
  - For overlapping intervals of one drug, the maximum active rate is used.
  - No eligible active interval is represented as zero recorded exposure.
  - Extreme and overlap flags are retained as separate features.
This is a retrospective research feature definition, not a claim of verified
physical infusion delivery at every instant.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "real" / "processed"
ANCHORS_FILE = PROCESSED / "mimic_features_strict.csv"
INTERVALS_FILE = PROCESSED / "vasopressor_intervals.csv"
OUTPUT_FILE = PROCESSED / "mimic_features_with_vasopressors.csv"
REPORT_FILE = PROCESSED / "vasopressor_feature_report.json"

MEDICATIONS = ("norepinephrine", "epinephrine", "dopamine", "dobutamine")
ELIGIBLE_STATUSES = {"ChangeDose/Rate", "FinishedRunning"}


def build_for_stay(anchors: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Sweep chronological anchors and infusion intervals for one ICU stay."""
    anchors = anchors.sort_values("_assessment_dt").copy()
    intervals = intervals.sort_values("_start_dt").copy()

    # Per medication: active records, min-end heap, and max-rate heap.
    states = {
        med: {"active": {}, "end_heap": [], "rate_heap": []}
        for med in MEDICATIONS
    }
    next_interval = 0
    interval_rows = list(
        intervals[
            ["_start_dt", "_end_dt", "medication", "rate", "extreme_rate_flag"]
        ].itertuples(index=False, name=None)
    )

    output = {f"vasopressor_{m}_rate": [] for m in MEDICATIONS}
    output.update({f"vasopressor_{m}_active": [] for m in MEDICATIONS})
    output.update({f"vasopressor_{m}_extreme_flag": [] for m in MEDICATIONS})
    output.update({f"vasopressor_{m}_overlap_flag": [] for m in MEDICATIONS})

    for assessment_time in anchors["_assessment_dt"]:
        # Add all eligible intervals that have started by this assessment.
        while next_interval < len(interval_rows):
            start, end, med, rate, extreme = interval_rows[next_interval]
            if start > assessment_time:
                break
            idx = next_interval
            next_interval += 1

            if med not in states or pd.isna(rate) or end <= assessment_time:
                continue

            state = states[med]
            rate = float(rate)
            is_extreme = bool(extreme) if not pd.isna(extreme) else False
            state["active"][idx] = (rate, end, is_extreme)
            heapq.heappush(state["end_heap"], (end, idx))
            heapq.heappush(state["rate_heap"], (-rate, idx))

        # Expire intervals at their exclusive end time.
        for med in MEDICATIONS:
            state = states[med]
            active = state["active"]
            end_heap = state["end_heap"]
            while end_heap and end_heap[0][0] <= assessment_time:
                _, idx = heapq.heappop(end_heap)
                active.pop(idx, None)

            rate_heap = state["rate_heap"]
            while rate_heap and rate_heap[0][1] not in active:
                heapq.heappop(rate_heap)

            count = len(active)
            if count:
                max_rate = -rate_heap[0][0] if rate_heap else 0.0
                extreme_active = any(v[2] for v in active.values())
            else:
                max_rate = 0.0
                extreme_active = False

            output[f"vasopressor_{med}_rate"].append(max_rate)
            output[f"vasopressor_{med}_active"].append(int(count > 0))
            output[f"vasopressor_{med}_extreme_flag"].append(int(extreme_active))
            output[f"vasopressor_{med}_overlap_flag"].append(int(count > 1))

    for col, values in output.items():
        anchors[col] = values
    return anchors


def main() -> None:
    if not ANCHORS_FILE.exists():
        raise FileNotFoundError(f"Missing strict feature file: {ANCHORS_FILE}")
    if not INTERVALS_FILE.exists():
        raise FileNotFoundError(f"Missing interval file: {INTERVALS_FILE}")

    print("Loading strict labelled anchors...")
    anchors = pd.read_csv(ANCHORS_FILE, low_memory=False)
    anchors["_assessment_dt"] = pd.to_datetime(anchors["assessment_time"], errors="raise")
    if anchors.duplicated(["stay_id", "assessment_time"]).any():
        raise ValueError("Duplicate stay_id/assessment_time keys in strict anchors")

    print("Loading vasopressor intervals...")
    intervals = pd.read_csv(
        INTERVALS_FILE,
        usecols=[
            "stay_id", "starttime", "endtime", "medication", "rate",
            "rateuom", "statusdescription", "extreme_rate_flag",
        ],
        low_memory=False,
    )
    intervals = intervals[
        intervals["medication"].isin(MEDICATIONS)
        & intervals["statusdescription"].isin(ELIGIBLE_STATUSES)
        & intervals["rateuom"].eq("mcg/kg/min")
    ].copy()
    intervals["_start_dt"] = pd.to_datetime(intervals["starttime"], errors="coerce")
    intervals["_end_dt"] = pd.to_datetime(intervals["endtime"], errors="coerce")
    intervals["rate"] = pd.to_numeric(intervals["rate"], errors="coerce")
    intervals["extreme_rate_flag"] = intervals["extreme_rate_flag"].fillna(False).astype(bool)
    intervals = intervals.dropna(subset=["stay_id", "_start_dt", "_end_dt", "rate"])
    intervals = intervals[intervals["_end_dt"] > intervals["_start_dt"]]
    intervals = intervals[intervals["rate"] > 0]

    # Only process stays represented in the strict labelled anchors.
    stay_ids = set(anchors["stay_id"].unique())
    intervals = intervals[intervals["stay_id"].isin(stay_ids)]
    interval_counts = intervals["medication"].value_counts().to_dict()

    print(f"Anchors: {len(anchors):,} | stays: {anchors.stay_id.nunique():,}")
    print(f"Eligible intervals in labelled stays: {len(intervals):,}")
    print("Building time-aligned features by ICU stay...")

    pieces = []
    grouped_intervals = {sid: grp for sid, grp in intervals.groupby("stay_id", sort=False)}
    for n, (stay_id, group) in enumerate(anchors.groupby("stay_id", sort=False), start=1):
        stay_intervals = grouped_intervals.get(stay_id)
        if stay_intervals is None:
            group = group.sort_values("assessment_time").copy()
            for med in MEDICATIONS:
                group[f"vasopressor_{med}_rate"] = 0.0
                group[f"vasopressor_{med}_active"] = 0
                group[f"vasopressor_{med}_extreme_flag"] = 0
                group[f"vasopressor_{med}_overlap_flag"] = 0
            pieces.append(group)
        else:
            pieces.append(build_for_stay(group, stay_intervals))
        if n % 500 == 0:
            print(f"  Processed {n:,}/{anchors.stay_id.nunique():,} stays")

    result = pd.concat(pieces, ignore_index=True)
    result = result.drop(columns=["_assessment_dt"], errors="ignore")
    result.to_csv(OUTPUT_FILE, index=False)

    report = {
        "input_anchors": str(ANCHORS_FILE),
        "input_intervals": str(INTERVALS_FILE),
        "output": str(OUTPUT_FILE),
        "anchor_rows": int(len(result)),
        "unique_patients": int(result["subject_id"].nunique()),
        "unique_stays": int(result["stay_id"].nunique()),
        "eligible_interval_counts_by_medication": {
            med: int(interval_counts.get(med, 0)) for med in MEDICATIONS
        },
        "eligible_statuses": sorted(ELIGIBLE_STATUSES),
        "rate_unit": "mcg/kg/min",
        "active_interval_rule": "starttime <= assessment_time < endtime",
        "overlap_rate_rule": "maximum active rate per medication",
        "no_active_interval_rate_value": 0,
        "clinical_interpretation": (
            "Retrospective research proxy based on eligible recorded intervals; "
            "not verification of physical infusion delivery."
        ),
        "feature_summary": {},
    }
    for med in MEDICATIONS:
        report["feature_summary"][med] = {
            "active_assessments": int(result[f"vasopressor_{med}_active"].sum()),
            "extreme_flag_assessments": int(result[f"vasopressor_{med}_extreme_flag"].sum()),
            "overlap_assessments": int(result[f"vasopressor_{med}_overlap_flag"].sum()),
            "maximum_rate": float(result[f"vasopressor_{med}_rate"].max()),
        }

    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Shape: {result.shape}")
    print(f"Report: {REPORT_FILE}")
    for med in MEDICATIONS:
        s = report["feature_summary"][med]
        print(
            f"{med}: active={s['active_assessments']:,}, "
            f"extreme={s['extreme_flag_assessments']:,}, "
            f"overlap={s['overlap_assessments']:,}, max_rate={s['maximum_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
