import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = {
    221906: ("Norepinephrine", 5),
    221289: ("Epinephrine", 5),
    221662: ("Dopamine", 40),
    221653: ("Dobutamine", 40),
}

cols = ["stay_id", "itemid", "starttime", "endtime",
        "rate", "rateuom", "patientweight"]

counts = {i: {"total": 0, "bad_weight": 0, "extreme": 0,
              "nonstandard": 0, "overlap_candidates": 0}
          for i in items}

for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
    df = chunk[chunk.itemid.isin(items)].copy()

    for item, (name, limit) in items.items():
        g = df[df.itemid == item]
        counts[item]["total"] += len(g)
        counts[item]["bad_weight"] += int((g.patientweight <= 1).sum())
        counts[item]["extreme"] += int(
            ((g.rateuom == "mcg/kg/min") & (g.rate > limit)).sum()
        )
        counts[item]["nonstandard"] += int(
            (g.rateuom != "mcg/kg/min").sum()
        )

print("========== AUDIT COUNTS ==========")
for item, (name, limit) in items.items():
    print(f"\n{name} (review threshold > {limit})")
    for key, value in counts[item].items():
        print(f"{key}: {value:,}")

print("\nThresholds are review flags, not automatic clinical exclusions.")
