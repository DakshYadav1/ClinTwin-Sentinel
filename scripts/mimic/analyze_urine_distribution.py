import pandas as pd
import numpy as np

PATH = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

LABELS = {
    226559: "Foley",
    226560: "Void",
    226627: "OR Urine",
    227488: "GU Irrigant Volume In",
    227489: "GU Irrigant/Urine Volume Out",
}

thresholds = [500, 1000, 2000, 5000, 10000, 20000, 50000]
values = {item: [] for item in LABELS}
top_records = {item: pd.DataFrame() for item in LABELS}
missing = {item: 0 for item in LABELS}
negative = {item: 0 for item in LABELS}
total = {item: 0 for item in LABELS}
above = {item: {t: 0 for t in thresholds} for item in LABELS}

columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "storetime", "itemid", "value"
]

for chunk in pd.read_csv(
    PATH, compression="gzip", usecols=columns,
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk = chunk[chunk["itemid"].isin(LABELS)].copy()

    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    for item in LABELS:
        part = chunk[chunk["itemid"] == item]
        if part.empty:
            continue

        total[item] += len(part)
        missing[item] += int(part["value"].isna().sum())

        valid = part["value"].dropna()
        negative[item] += int((valid < 0).sum())

        values[item].extend(valid.tolist())

        for t in thresholds:
            above[item][t] += int((valid > t).sum())

        largest = part.nlargest(20, "value")
        top_records[item] = (
            pd.concat([top_records[item], largest])
            .nlargest(20, "value")
        )

print("\n========== URINE OUTPUT DISTRIBUTION ==========")

for item, label in LABELS.items():
    arr = np.asarray(values[item], dtype=float)

    print(f"\n{'=' * 55}")
    print(f"{label} | itemid={item}")
    print("=" * 55)
    print("Total records:", total[item])
    print("Missing values:", missing[item])
    print("Negative values:", negative[item])

    if len(arr):
        print("\nPercentiles:")
        for p in [0, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:
            print(f"  {p:5.1f}%: {np.percentile(arr, p):,.2f} mL")

        print("\nValues above thresholds:")
        for t in thresholds:
            print(f"  > {t:>6} mL: {above[item][t]:,}")

        print("\nTOP 10 RECORDS:")
        print(
            top_records[item].nlargest(10, "value")[
                ["subject_id", "hadm_id", "stay_id",
                 "charttime", "storetime", "value"]
            ].to_string(index=False)
        )
