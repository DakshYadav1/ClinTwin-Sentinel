import pandas as pd

source = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221653: "dobutamine",
}

columns = [
    "subject_id", "hadm_id", "stay_id", "starttime", "endtime",
    "itemid", "amount", "amountuom", "rate", "rateuom",
    "patientweight", "statusdescription", "orderid", "linkorderid"
]

parts = []

for chunk in pd.read_csv(source, usecols=columns, chunksize=500_000):
    df = chunk[
        chunk["itemid"].isin(items)
        & (pd.to_numeric(chunk["patientweight"], errors="coerce") > 1)
        & chunk["rateuom"].eq("mcg/kg/min")
        & chunk["statusdescription"].ne("Bolus")
    ].copy()

    if not df.empty:
        parts.append(df)

df = pd.concat(parts, ignore_index=True)
df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")
df["duration_min"] = (
    df["endtime"] - df["starttime"]
).dt.total_seconds() / 60

print("Loaded valid infusion rows:", len(df))

for itemid, medication in items.items():
    med = df[df["itemid"] == itemid].copy()
    print(f"\n========== {medication.upper()} ==========")

    overlap_examples = []

    for stay_id, group in med.groupby("stay_id"):
        group = group.sort_values(["starttime", "endtime"])

        previous = []
        for row in group.itertuples():
            active = [
                p for p in previous
                if p.endtime > row.starttime
            ]

            for p in active:
                overlap_examples.append((p, row))

            previous.append(row)

        if len(overlap_examples) >= 3:
            break

    print("Sample overlapping pairs:")
    for previous, current in overlap_examples[:3]:
        print({
            "stay_id": current.stay_id,
            "previous": {
                "start": str(previous.starttime),
                "end": str(previous.endtime),
                "rate": previous.rate,
                "status": previous.statusdescription,
                "orderid": previous.orderid,
                "linkorderid": previous.linkorderid,
            },
            "current": {
                "start": str(current.starttime),
                "end": str(current.endtime),
                "rate": current.rate,
                "status": current.statusdescription,
                "orderid": current.orderid,
                "linkorderid": current.linkorderid,
            }
        })

    long_rows = med[med["duration_min"] > 1440].head(3)
    print("\nSample intervals longer than 24 hours:")
    print(long_rows[
        ["stay_id", "starttime", "endtime", "duration_min",
         "rate", "statusdescription", "orderid", "linkorderid"]
    ].to_string(index=False))
