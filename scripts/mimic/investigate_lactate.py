import pandas as pd

FILE = "data/real/processed/suspicious_lab_source_records.csv"

df = pd.read_csv(FILE)

target = df[
    (df["itemid"] == 50813) &
    (df["valuenum"] > 100000)
]

print("Exceptional lactate records:")
print(
    target[
        [
            "labevent_id",
            "value",
            "valuenum",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
            "comments"
        ]
    ].to_string(index=False)
)

print("\nNumber of exceptional lactate records:", len(target))
