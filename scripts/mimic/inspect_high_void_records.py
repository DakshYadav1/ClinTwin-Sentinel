import pandas as pd

PATH = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

rows = []

for chunk in pd.read_csv(
    PATH,
    compression="gzip",
    usecols=[
        "subject_id", "hadm_id", "stay_id",
        "charttime", "storetime", "itemid", "value"
    ],
    chunksize=1_000_000
):
    chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")

    selected = chunk[
        (chunk["itemid"] == 226560) &
        (chunk["value"] > 2000)
    ]

    if not selected.empty:
        rows.append(selected)

result = pd.concat(rows, ignore_index=True)
result = result.sort_values("value", ascending=False)

print("HIGH VOID RECORDS:", len(result))
print(
    result.to_string(index=False)
)
