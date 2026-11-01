import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = {
    221906: "Norepinephrine",
    221289: "Epinephrine",
    221662: "Dopamine",
    221653: "Dobutamine",
}

cols = [
    "subject_id", "stay_id", "starttime", "endtime",
    "itemid", "rate", "rateuom", "patientweight",
    "statusdescription"
]

frames = []

for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
    df = chunk[chunk["itemid"].isin(items)].copy()
    frames.append(df)

df = pd.concat(frames, ignore_index=True)
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

print("========== RATE DISTRIBUTION ==========")

for item, name in items.items():
    g = df[df.itemid == item]
    valid = g[g.rateuom == "mcg/kg/min"]["rate"].dropna()

    print(f"\n{name}: {len(valid):,} standardized records")
    print(valid.quantile([0.50, 0.90, 0.95, 0.99, 0.999]).to_string())
    print("Above 1:", (valid > 1).sum())
    print("Above 5:", (valid > 5).sum())
    print("Above 10:", (valid > 10).sum())
    print("Missing weight:", g.patientweight.isna().sum())

print("\n========== NONSTANDARD UNITS ==========")
print(df[df.rateuom != "mcg/kg/min"][
    ["itemid", "rate", "rateuom", "patientweight", "statusdescription"]
].head(20).to_string(index=False))

print("\n========== EXTREME RATE EXAMPLES ==========")

for item, name in items.items():
    g = df[(df.itemid == item) & (df.rateuom == "mcg/kg/min")]
    print(f"\n{name}")
    print(g.nlargest(5, "rate")[[
        "stay_id", "starttime", "endtime", "rate",
        "patientweight", "statusdescription"
    ]].to_string(index=False))
