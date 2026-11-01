import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz"

items = [221906, 221289, 221662, 221653]

cols = [
    "subject_id", "hadm_id", "stay_id", "starttime",
    "endtime", "itemid", "amount", "amountuom",
    "rate", "rateuom", "patientweight", "statusdescription"
]

targets = {
    32385234, 39323481, 37996113, 33017235,
    30241837, 33181880, 32556479, 33341000,
    33994081, 39017077, 36740821, 30680941,
    31908350, 36471889, 33445309, 30549305,
    38121875, 31241790
}

frames = []

for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
    mask = (
        chunk["stay_id"].isin(targets) |
        chunk["patientweight"].eq(1) |
        (
            chunk["itemid"].eq(221906) &
            chunk["rateuom"].eq("mg/kg/min")
        )
    ) & chunk["itemid"].isin(items)

    if mask.any():
        frames.append(chunk.loc[mask])

df = pd.concat(frames, ignore_index=True)

print("Suspicious records:", len(df))
print(df.sort_values(["stay_id", "starttime"]).to_string(index=False))
