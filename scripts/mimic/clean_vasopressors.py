import pandas as pd
import json

source = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"
output = "data/real/processed/cleaned_vasopressors.csv"
report_path = "data/real/processed/vasopressor_cleaning_report.json"

items = {
    221906: ("norepinephrine", 5),
    221289: ("epinephrine", 5),
    221662: ("dopamine", 40),
    221653: ("dobutamine", 40),
}

columns = [
    "subject_id", "hadm_id", "stay_id", "starttime", "endtime",
    "itemid", "amount", "amountuom", "rate", "rateuom",
    "patientweight", "statusdescription"
]

counts = {
    "selected": 0,
    "retained": 0,
    "invalid_weight_excluded": 0,
    "nonstandard_unit_excluded": 0,
    "extreme_rate_flagged": 0,
    "by_medication": {}
}

first = True

for chunk in pd.read_csv(source, usecols=columns, chunksize=500_000):
    df = chunk[chunk.itemid.isin(items)].copy()

    if df.empty:
        continue

    counts["selected"] += len(df)

    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["patientweight"] = pd.to_numeric(
        df["patientweight"], errors="coerce"
    )

    valid_weight = df["patientweight"] > 1
    valid_unit = df["rateuom"].eq("mcg/kg/min")

    counts["invalid_weight_excluded"] += int((~valid_weight).sum())
    counts["nonstandard_unit_excluded"] += int(
        (valid_weight & ~valid_unit).sum()
    )

    df = df[valid_weight & valid_unit].copy()

    if df.empty:
        continue

    df["medication"] = df["itemid"].map(
        {i: name for i, (name, _) in items.items()}
    )

    df["extreme_rate_flag"] = df.apply(
        lambda row: row["rate"] > items[int(row["itemid"])][1],
        axis=1
    )

    counts["extreme_rate_flagged"] += int(
        df["extreme_rate_flag"].sum()
    )
    counts["retained"] += len(df)

    for item, (name, _) in items.items():
        n = int((df["itemid"] == item).sum())
        counts["by_medication"][name] = (
            counts["by_medication"].get(name, 0) + n
        )

    df.to_csv(
        output,
        mode="w" if first else "a",
        header=first,
        index=False
    )
    first = False

with open(report_path, "w") as f:
    json.dump(counts, f, indent=2)

print(json.dumps(counts, indent=2))
print("\nSaved:", output)
print("Report:", report_path)
