import gzip

file = "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz"

print("Checking gzip integrity...")

with gzip.open(file, "rb") as f:
    total = 0
    while True:
        block = f.read(8 * 1024 * 1024)
        if not block:
            break
        total += len(block)

print("GZIP integrity: PASS")
print(f"Uncompressed bytes: {total:,}")
