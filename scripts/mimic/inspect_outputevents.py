import pandas as pd

path = "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz"

df = pd.read_csv(path, compression="gzip", nrows=5)

print("COLUMNS:")
print(df.columns.tolist())

print("\nSAMPLE RECORDS:")
print(df.to_string(index=False))
