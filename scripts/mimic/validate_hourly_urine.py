import pandas as pd

path = "data/real/processed/hourly_urine_output.csv"

chunks = pd.read_csv(path, chunksize=500_000)

total_rows = 0
both_sources = 0
missing_foley = 0
missing_void = 0
extreme_500 = 0
extreme_1000 = 0
extreme_5000 = 0
maximum = 0
extreme_rows = []

for chunk in chunks:
    total_rows += len(chunk)

    both_sources += int(
        (chunk["foley_output_ml"].notna() &
         chunk["void_output_ml"].notna()).sum()
    )

    missing_foley += int(chunk["foley_output_ml"].isna().sum())
    missing_void += int(chunk["void_output_ml"].isna().sum())

    values = chunk["total_recorded_urine_ml"]

    extreme_500 += int((values > 500).sum())
    extreme_1000 += int((values > 1000).sum())
    extreme_5000 += int((values > 5000).sum())

    maximum = max(maximum, values.max())

    selected = chunk[values > 5000]
    if not selected.empty:
        extreme_rows.append(selected)

print("TOTAL HOURLY ROWS:", total_rows)
print("BOTH FOLEY AND VOID:", both_sources)
print("FOLEY MISSING:", missing_foley)
print("VOID MISSING:", missing_void)
print("VOLUME >500 mL:", extreme_500)
print("VOLUME >1000 mL:", extreme_1000)
print("VOLUME >5000 mL:", extreme_5000)
print("MAXIMUM HOURLY VOLUME:", maximum)

if extreme_rows:
    print("\nEXTREME RECORDS:")
    print(
        pd.concat(extreme_rows)
        .sort_values("total_recorded_urine_ml", ascending=False)
        .head(20)
        .to_string(index=False)
    )
