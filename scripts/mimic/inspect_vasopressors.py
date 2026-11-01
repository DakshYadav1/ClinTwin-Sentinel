import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = {
    221906: "Norepinephrine",
    221289: "Epinephrine",
    221662: "Dopamine",
    221653: "Dobutamine",
    222315: "Vasopressin",
}

columns = [
    "subject_id", "hadm_id", "stay_id", "starttime",
    "endtime", "itemid", "amount", "amountuom",
    "rate", "rateuom", "patientweight", "statusdescription"
]

counts = {}
units = {}
rate_ranges = {}
samples = []

print("Scanning inputevents...")

for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
    df = chunk[chunk["itemid"].isin(items)].copy()

    for item, group in df.groupby("itemid"):
        item = int(item)
        counts[item] = counts.get(item, 0) + len(group)

        for col in ["amountuom", "rateuom", "statusdescription"]:
            for value, n in group[col].fillna("MISSING").value_counts().items():
                key = (item, col, str(value))
                units[key] = units.get(key, 0) + int(n)

        valid_rates = pd.to_numeric(group["rate"], errors="coerce").dropna()
        if len(valid_rates):
            old = rate_ranges.get(item, [float("inf"), float("-inf")])
            rate_ranges[item] = [
                min(old[0], valid_rates.min()),
                max(old[1], valid_rates.max())
            ]

        if len(samples) < 30:
            samples.append(group.head(30 - len(samples)))

print("\n========== VASOPRESSOR SUMMARY ==========")

for item, name in items.items():
    print(f"\n{name} ({item})")
    print("Records:", counts.get(item, 0))
    print("Rate range:", rate_ranges.get(item, "No numeric rates"))

    for (i, col, value), n in units.items():
        if i == item:
            print(f"  {col} = {value}: {n:,}")

if samples:
    print("\n========== SAMPLE RECORDS ==========")
    print(pd.concat(samples).to_string(index=False))

print("\nVasopressor inspection complete.")
