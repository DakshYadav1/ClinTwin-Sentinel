import pandas as pd

icu_path = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"
lab_path = "data/real/raw/mimiciv-3.1/hosp/labevents.csv.gz"

print("\n========== FiO2 DISTRIBUTION ==========")

bins = {
    "zero_or_negative": 0,
    "fraction_0_to_1": 0,
    "above_1_to_below_20": 0,
    "percentage_20_to_100": 0,
    "above_100": 0,
    "missing": 0,
}

fio_samples = []
fio_values = []

for chunk in pd.read_csv(
    icu_path,
    usecols=["subject_id", "hadm_id", "stay_id", "charttime",
             "itemid", "valuenum"],
    chunksize=1_000_000
):
    df = chunk[chunk["itemid"] == 223835].copy()
    if df.empty:
        continue

    v = df["valuenum"]
    bins["zero_or_negative"] += int((v <= 0).sum())
    bins["fraction_0_to_1"] += int(((v > 0) & (v <= 1)).sum())
    bins["above_1_to_below_20"] += int(((v > 1) & (v < 20)).sum())
    bins["percentage_20_to_100"] += int(((v >= 20) & (v <= 100)).sum())
    bins["above_100"] += int((v > 100).sum())
    bins["missing"] += int(v.isna().sum())

    valid = v.dropna()
    fio_values.extend(valid.sample(min(1000, len(valid))).tolist())

    extremes = df[(v > 100) | (v <= 0)]
    if len(extremes):
        fio_samples.append(extremes.head(5))

print("FiO2 value groups:", bins)
print("Sample of observed FiO2 values:", sorted(set(fio_values))[:60])

if fio_samples:
    print("\nSuspicious FiO2 records:")
    print(pd.concat(fio_samples).head(20).to_string(index=False))

print("\n========== PaO2 DISTRIBUTION ==========")

pao_values = []
pao_extremes = []

for chunk in pd.read_csv(
    lab_path,
    usecols=["labevent_id", "subject_id", "hadm_id",
             "charttime", "itemid", "valuenum", "valueuom"],
    chunksize=1_000_000
):
    df = chunk[chunk["itemid"] == 50821].copy()
    if df.empty:
        continue

    values = df["valuenum"].dropna()
    if len(values):
        pao_values.extend(values.sample(min(5000, len(values))).tolist())

    extremes = df[(df["valuenum"] < 20) | (df["valuenum"] > 500)]
    if len(extremes):
        pao_extremes.append(extremes.head(5))

if pao_values:
    print("Sampled PaO2 percentiles:")
    print(pd.Series(pao_values).quantile(
        [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    ).to_string())

if pao_extremes:
    print("\nSuspicious PaO2 records:")
    print(pd.concat(pao_extremes).head(20).to_string(index=False))

print("\nInvestigation complete.")
