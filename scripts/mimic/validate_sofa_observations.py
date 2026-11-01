import pandas as pd

icu_path = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"
lab_path = "data/real/raw/mimiciv-3.1/hosp/labevents.csv.gz"

icu_items = {
    223835: "Inspired O2 Fraction",
    220739: "GCS Eye Opening",
    223900: "GCS Verbal Response",
    223901: "GCS Motor Response",
}

stats = {
    item: {"count": 0, "units": {}, "min": float("inf"),
           "max": float("-inf"), "samples": set()}
    for item in icu_items
}

print("\nScanning chartevents for FiO2 and GCS...")

for chunk in pd.read_csv(
    icu_path,
    usecols=["itemid", "valuenum", "valueuom"],
    chunksize=1_000_000
):
    chunk = chunk[chunk["itemid"].isin(icu_items)]

    for item, group in chunk.groupby("itemid"):
        s = stats[int(item)]
        values = group["valuenum"].dropna()

        s["count"] += len(group)
        s["min"] = min(s["min"], values.min()) if len(values) else s["min"]
        s["max"] = max(s["max"], values.max()) if len(values) else s["max"]

        for unit, count in group["valueuom"].fillna("MISSING").value_counts().items():
            s["units"][str(unit)] = s["units"].get(str(unit), 0) + int(count)

        s["samples"].update(values.unique().tolist())

for item, name in icu_items.items():
    s = stats[item]
    print(f"\n--- {name} ({item}) ---")
    print("Records:", f"{s['count']:,}")
    print("Units:", s["units"])
    print("Minimum:", s["min"] if s["count"] else "N/A")
    print("Maximum:", s["max"] if s["count"] else "N/A")
    print("Sample values:", sorted(s["samples"])[:35])

print("\nScanning laboratory records for PaO2...")

count = 0
units = {}
minimum = float("inf")
maximum = float("-inf")
samples = set()

for chunk in pd.read_csv(
    lab_path,
    usecols=["itemid", "valuenum", "valueuom"],
    chunksize=1_000_000
):
    group = chunk[chunk["itemid"] == 50821]
    if group.empty:
        continue

    values = group["valuenum"].dropna()
    count += len(group)

    if len(values):
        minimum = min(minimum, values.min())
        maximum = max(maximum, values.max())
        samples.update(values.unique().tolist())

    for unit, n in group["valueuom"].fillna("MISSING").value_counts().items():
        units[str(unit)] = units.get(str(unit), 0) + int(n)

print("\n--- PaO2 (50821) ---")
print("Records:", f"{count:,}")
print("Units:", units)
print("Minimum:", minimum if count else "N/A")
print("Maximum:", maximum if count else "N/A")
print("Sample values:", sorted(samples)[:35])

print("\nSOFA source validation scan complete.")
