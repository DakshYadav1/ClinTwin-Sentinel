import pandas as pd
import numpy as np
import heapq
import json
from collections import defaultdict

MAP_FILE = "data/real/processed/hourly_map.csv"
INTERVAL_FILE = "data/real/processed/vasopressor_intervals.csv"
OUTPUT = "data/real/processed/hourly_cardiovascular_sofa.csv"
REPORT = "data/real/processed/cardiovascular_sofa_report.json"

MEDS = ["norepinephrine", "epinephrine", "dopamine", "dobutamine"]

print("Loading hourly MAP assessments...")
base = pd.read_csv(MAP_FILE, parse_dates=["assessment_time"])
base = base.sort_values(
    ["stay_id", "assessment_time"]
).reset_index(drop=True)

print("Loading vasopressor intervals...")
iv = pd.read_csv(
    INTERVAL_FILE,
    usecols=["stay_id", "starttime", "endtime", "rate", "medication"],
    parse_dates=["starttime", "endtime"]
)

iv = iv[iv["medication"].isin(MEDS)].copy()
iv = iv.sort_values(["stay_id", "starttime"])

interval_groups = defaultdict(dict)

for (stay_id, med), group in iv.groupby(["stay_id", "medication"]):
    interval_groups[int(stay_id)][med] = (
        group["starttime"].astype("int64").to_numpy(),
        group["endtime"].astype("int64").to_numpy(),
        group["rate"].to_numpy(dtype=np.float64)
    )

# Numeric output arrays avoid millions of temporary dictionaries.
dose_arrays = {
    med: np.zeros(len(base), dtype=np.float32)
    for med in MEDS
}
overlap_array = np.zeros(len(base), dtype=bool)

print("Calculating active medication doses...")

for stay_id, group in base.groupby("stay_id", sort=False):
    positions = group.index.to_numpy()
    times = group["assessment_time"].astype("int64").to_numpy()

    stay_intervals = interval_groups.get(int(stay_id), {})
    states = {}

    for med in MEDS:
        if med in stay_intervals:
            starts, ends, rates = stay_intervals[med]
        else:
            starts = np.array([], dtype=np.int64)
            ends = np.array([], dtype=np.int64)
            rates = np.array([], dtype=np.float64)

        states[med] = {
            "starts": starts,
            "ends": ends,
            "rates": rates,
            "pointer": 0,
            "max_heap": [],
            "end_heap": []
        }

    for pos, t in zip(positions, times):
        for med in MEDS:
            state = states[med]
            starts = state["starts"]
            ends = state["ends"]
            rates = state["rates"]
            p = state["pointer"]

            while p < len(starts) and starts[p] <= t:
                heapq.heappush(
                    state["max_heap"],
                    (-rates[p], ends[p], p)
                )
                heapq.heappush(state["end_heap"], ends[p])
                p += 1

            state["pointer"] = p

            while state["end_heap"] and state["end_heap"][0] <= t:
                heapq.heappop(state["end_heap"])

            while (
                state["max_heap"]
                and state["max_heap"][0][1] <= t
            ):
                heapq.heappop(state["max_heap"])

            if state["end_heap"]:
                if len(state["end_heap"]) > 1:
                    overlap_array[pos] = True

                if state["max_heap"]:
                    dose_arrays[med][pos] = (
                        -state["max_heap"][0][0]
                    )

    if len(positions) % 10000 == 0:
        print("Processed stay:", stay_id)

for med in MEDS:
    base["active_" + med] = dose_arrays[med]

base["vasopressor_overlap"] = overlap_array

# Original SOFA cardiovascular dose thresholds.
dopamine = base["active_dopamine"].to_numpy()
dobutamine = base["active_dobutamine"].to_numpy()
norepi = base["active_norepinephrine"].to_numpy()
epi = base["active_epinephrine"].to_numpy()
map_values = base["map_mmhg"].to_numpy()

has_pressor = (
    (dopamine > 0)
    | (dobutamine > 0)
    | (norepi > 0)
    | (epi > 0)
)

scores = np.full(len(base), np.nan, dtype=np.float32)

# MAP-based score when no qualifying vasopressor is active.
no_pressor_with_map = (~has_pressor) & np.isfinite(map_values)
scores[no_pressor_with_map] = (
    map_values[no_pressor_with_map] < 70
).astype(np.float32)

# Medication-based scoring.
scores[has_pressor] = 0

scores[has_pressor & (dopamine > 0) & (dopamine <= 5)] = 2
scores[has_pressor & (dobutamine > 0)] = np.maximum(
    scores[has_pressor & (dobutamine > 0)], 2
)

scores[has_pressor & (dopamine > 5) & (dopamine <= 15)] = 3

scores[
    has_pressor
    & (
        ((norepi > 0) & (norepi <= 0.1))
        | ((epi > 0) & (epi <= 0.1))
    )
] = np.maximum(
    scores[
        has_pressor
        & (
            ((norepi > 0) & (norepi <= 0.1))
            | ((epi > 0) & (epi <= 0.1))
        )
    ],
    3
)

scores[
    has_pressor
    & (
        (dopamine > 15)
        | (norepi > 0.1)
        | (epi > 0.1)
    )
] = 4

base["cardiovascular_sofa"] = scores

stats = {
    "assessments": len(base),
    "with_map": int(base["map_mmhg"].notna().sum()),
    "missing_map": int(base["map_mmhg"].isna().sum()),
    "with_active_vasopressor": int(has_pressor.sum()),
    "overlap_assessments": int(overlap_array.sum()),
    "missing_cv_score": int(np.isnan(scores).sum()),
    "map_missing_but_vasopressor_active": int(
        ((~np.isfinite(map_values)) & has_pressor).sum()
    ),
    "score_distribution": {
        str(int(k)): int(v)
        for k, v in pd.Series(scores).value_counts().sort_index().items()
        if pd.notna(k)
    }
}

base.to_csv(OUTPUT, index=False)

with open(REPORT, "w") as f:
    json.dump(stats, f, indent=2)

print("\n========== CARDIOVASCULAR SOFA ==========")
print(json.dumps(stats, indent=2))
print("\nSaved:", OUTPUT)
print("Report:", REPORT)
