import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"
ids = [226559, 227488, 227489]

df = pd.read_csv(
    path,
    compression="gzip",
    usecols=["subject_id", "hadm_id", "stay_id",
             "charttime", "storetime", "itemid", "value"],
    dtype={"itemid": "int32"}
)

df = df[df["itemid"].isin(ids)].copy()
df["charttime"] = pd.to_datetime(df["charttime"])
df["hour"] = df["charttime"].dt.floor("h")

grouped = df.groupby(
    ["stay_id", "hour", "itemid"]
)["value"].agg(["count", "sum"]).unstack("itemid", fill_value=0)

for item in ids:
    if ("sum", item) not in grouped.columns:
        grouped[("sum", item)] = 0

overlap = grouped[
    (grouped[("count", 226559)] > 0) &
    (grouped[("count", 227489)] > 0)
]

print("Foley + irrigation-output overlapping stay-hours:", len(overlap))
print("\nSAMPLE OVERLAPPING HOURS:")
print(overlap.head(10).to_string())

sample_keys = overlap.head(5).index
print("\nRAW RECORDS FOR FIVE OVERLAPPING HOURS:")

for stay_id, hour in sample_keys:
    rows = df[
        (df["stay_id"] == stay_id) &
        (df["hour"] == hour)
    ].sort_values(["charttime", "itemid"])

    print("\nStay:", stay_id, "| Hour:", hour)
    print(rows[
        ["charttime", "storetime", "itemid", "value"]
    ].to_string(index=False))
