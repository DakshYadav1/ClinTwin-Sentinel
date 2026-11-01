import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"

BOUNDS = {
    220045: (20, 250),  # Heart rate
    220210: (1, 100),   # Respiratory rate
    223761: (77, 113),  # Temperature Fahrenheit
    223762: (25, 45),   # Temperature Celsius
    220050: (40, 300),  # Arterial systolic BP
    220051: (10, 200),  # Arterial diastolic BP
    220052: (20, 250),  # Arterial mean BP
    220179: (40, 300),  # NIBP systolic BP
    220180: (10, 200),  # NIBP diastolic BP
    220181: (20, 250),  # NIBP mean BP
    224688: (1, 100),   # Respiratory rate set
    224689: (1, 100),   # Spontaneous respiratory rate
    224690: (1, 100),   # Total respiratory rate
    229841: (21, 100)   # FiO2 percentage
}

stats = {
    item: {
        "count": 0,
        "missing": 0,
        "min": float("inf"),
        "max": float("-inf"),
        "below_bound": 0,
        "above_bound": 0,
        "sample": []
    }
    for item in BOUNDS
}

for chunk in pd.read_csv(
    PATH,
    compression="gzip",
    usecols=["itemid", "valuenum"],
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")

    chunk = chunk[chunk["itemid"].isin(BOUNDS)]

    for itemid, group in chunk.groupby("itemid"):
        itemid = int(itemid)
        values = group["valuenum"]
        valid = values.dropna()

        s = stats[itemid]
        low, high = BOUNDS[itemid]

        s["count"] += len(group)
        s["missing"] += int(values.isna().sum())

        if not valid.empty:
            s["min"] = min(s["min"], float(valid.min()))
            s["max"] = max(s["max"], float(valid.max()))
            s["below_bound"] += int((valid < low).sum())
            s["above_bound"] += int((valid > high).sum())

            sample = valid.sample(
                n=min(2000, len(valid)),
                random_state=42
            ).tolist()
            s["sample"].extend(sample)

print("\n========== VITAL-SIGN VALUE DISTRIBUTIONS ==========")

for itemid, s in stats.items():
    print(f"\nItem ID: {itemid}")
    print("Records:", s["count"])
    print("Missing numeric values:", s["missing"])

    if s["count"] == 0 or s["min"] == float("inf"):
        print("No numeric observations")
        continue

    print("Minimum:", s["min"])
    print("Maximum:", s["max"])

    sample = pd.Series(s["sample"])
    print("Approximate sample percentiles:")
    print(sample.quantile([0.01, 0.25, 0.50, 0.75, 0.99]).to_string())

    print("Below screening range:", s["below_bound"])
    print("Above screening range:", s["above_bound"])
    print("Screening range:", BOUNDS[itemid])

print("\nNOTE: Screening ranges are for data-quality review only.")
print("They do not automatically determine which clinical values are excluded.")
