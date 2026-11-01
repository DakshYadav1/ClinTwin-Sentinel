import pandas as pd

source = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221653: "dobutamine",
}

columns = [
    "stay_id", "starttime", "endtime", "itemid",
    "rate", "statusdescription", "orderid", "linkorderid",
    "patientweight", "rateuom"
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
df["starttime"] = pd.to_datetime(df["starttime"])
df["endtime"] = pd.to_datetime(df["endtime"])

print("========== OVERLAP CLASSIFICATION ==========")

for itemid, medication in items.items():
    med = df[df["itemid"] == itemid]
    same_link = 0
    different_link = 0
    examples = []

    for stay_id, group in med.groupby("stay_id"):
        group = group.sort_values(["starttime", "endtime"])
        active = []

        for row in group.itertuples():
            active = [p for p in active if p.endtime > row.starttime]

            for p in active:
                if p.linkorderid == row.linkorderid:
                    same_link += 1
                    category = "SAME LINK"
                else:
                    different_link += 1
                    category = "DIFFERENT LINK"

                if len(examples) < 2:
                    examples.append((
                        category, stay_id,
                        p.starttime, p.endtime, p.rate,
                        row.starttime, row.endtime, row.rate,
                        p.linkorderid, row.linkorderid
                    ))

            active.append(row)

    print(f"\n{medication}:")
    print("Same linkorderid overlaps:", same_link)
    print("Different linkorderid overlaps:", different_link)

    for example in examples:
        print("Example:", example)

print("\n========== LONG INTERVAL STATUS ==========")
df["duration_hours"] = (
    df["endtime"] - df["starttime"]
).dt.total_seconds() / 3600

long = df[df["duration_hours"] > 24]

print("Intervals over 24 hours:", len(long))
print(long.groupby(
    ["itemid", "statusdescription"]
).size().to_string())

print("\nLong intervals sharing linkorderid with another row:")
long_keys = set(zip(long["stay_id"], long["linkorderid"]))

linked_counts = df.groupby(
    ["stay_id", "linkorderid"]
).size()

print(sum(linked_counts.get(key, 0) > 1 for key in long_keys))
