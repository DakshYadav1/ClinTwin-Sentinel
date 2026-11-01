import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

targets = {
    33010863: "2141-09-11 06:00:00",  # Extreme void
    33676353: "2145-02-08 15:43:00",  # Foley 30331
    34317198: "2149-05-07 12:35:00",  # Foley 25025
    31127752: "2125-01-13 20:45:00",  # Foley 8000
    38986983: "2125-06-06 00:00:00",  # Foley 7030
    33292306: "2134-04-07 06:05:00",  # Foley 6090
    37896702: "2123-05-20 20:48:00",  # Foley 5745
    32004192: "2110-11-06 17:45:00",  # Foley 5400
    34364406: "2144-11-15 06:00:00",  # Foley 5300
    37782301: "2115-11-10 21:30:00",  # Foley 5100
    33204925: "2154-06-14 06:00:00",  # Foley 5000
    38400142: "2179-03-08 02:00:00",  # Irrigation episode
}

itemids = [226559, 226560, 226627, 227488, 227489]
columns = [
    "subject_id", "hadm_id", "stay_id",
    "charttime", "storetime", "itemid", "value"
]

records = []
negative_records = []

for chunk in pd.read_csv(
    PATH, compression="gzip", usecols=columns,
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    selected = chunk[
        chunk["stay_id"].isin(targets) &
        chunk["itemid"].isin(itemids)
    ].copy()

    if not selected.empty:
        selected["charttime"] = pd.to_datetime(selected["charttime"])
        selected["storetime"] = pd.to_datetime(selected["storetime"])

        for stay_id, target_time in targets.items():
            center = pd.Timestamp(target_time)
            part = selected[
                (selected["stay_id"] == stay_id) &
                (selected["charttime"] >= center - pd.Timedelta(hours=6)) &
                (selected["charttime"] <= center + pd.Timedelta(hours=6))
            ]
            if not part.empty:
                records.append(part)

    neg = chunk[
        chunk["itemid"].isin([226559, 226560]) &
        (chunk["value"] < 0)
    ]
    if not neg.empty:
        negative_records.append(neg)

print("\n========== EXTREME RECORDS: SIX-HOUR CONTEXT ==========")

if records:
    result = pd.concat(records).drop_duplicates()
    result = result.sort_values(["stay_id", "charttime", "itemid"])

    for stay_id, group in result.groupby("stay_id"):
        print(f"\nSTAY {stay_id} | TARGET {targets[stay_id]}")
        print(group[
            ["charttime", "storetime", "itemid", "value"]
        ].to_string(index=False))

print("\n========== NEGATIVE URINE VALUES ==========")
if negative_records:
    negatives = pd.concat(negative_records).drop_duplicates()
    print(negatives.head(30).to_string(index=False))
