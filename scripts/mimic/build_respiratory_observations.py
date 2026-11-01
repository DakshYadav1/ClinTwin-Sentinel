
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/real/raw/mimiciv-3.1"
PROCESSED = ROOT / "data/real/processed"

COHORT_FILE = PROCESSED / "strict_icu_cohort.csv"
LAB_FILE = RAW / "hosp/labevents.csv.gz"
CHART_FILE = RAW / "icu/chartevents.csv.gz"

PAO2_FILE = PROCESSED / "combined_pao2.csv"
FIO2_FILE = PROCESSED / "cleaned_fio2.csv"
PAIRED_FILE = PROCESSED / "paired_pao2_fio2.csv"
REPORT_FILE = PROCESSED / "respiratory_observation_report.json"

PAO2_ID = 50821
FIO2_ID = 223835
CHUNK_SIZE = 500_000
PAIR_TOLERANCE = pd.Timedelta(minutes=60)


def load_cohort():
    cohort = pd.read_csv(
        COHORT_FILE,
        parse_dates=["intime", "outtime"]
    )
    return cohort


def extract_pao2(cohort):
    print("Extracting PaO2...")
    cohort_map = cohort[
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    ]

    output = []
    total = 0

    columns = [
        "labevent_id", "subject_id", "hadm_id",
        "itemid", "charttime", "valuenum", "valueuom"
    ]

    for chunk in pd.read_csv(
        LAB_FILE, compression="gzip",
        usecols=columns, chunksize=CHUNK_SIZE
    ):
        chunk = chunk[chunk["itemid"] == PAO2_ID]
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(
            chunk["charttime"], errors="coerce"
        )
        chunk["valuenum"] = pd.to_numeric(
            chunk["valuenum"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["subject_id", "hadm_id", "charttime", "valuenum"]
        )

        chunk = chunk.merge(
            cohort_map,
            on=["subject_id", "hadm_id"],
            how="inner"
        )

        chunk = chunk[
            (chunk["charttime"] >= chunk["intime"]) &
            (chunk["charttime"] <= chunk["outtime"]) &
            (chunk["valuenum"] > 0) &
            (chunk["valuenum"] <= 600)
        ].copy()

        if not chunk.empty:
            chunk["pao2_mmhg"] = chunk["valuenum"]
            chunk["pao2_extreme_flag"] = (
                chunk["pao2_mmhg"] > 500
            ).astype("int8")

            output.append(
                chunk[[
                    "labevent_id", "subject_id", "hadm_id",
                    "stay_id", "charttime", "pao2_mmhg",
                    "pao2_extreme_flag"
                ]]
            )
            total += len(chunk)

        print(f"PaO2 records retained: {total:,}")

    result = pd.concat(output, ignore_index=True)
    result = result.drop_duplicates(
        subset=["stay_id", "charttime", "pao2_mmhg"]
    )
    result.to_csv(PAO2_FILE, index=False)

    print(f"Saved {len(result):,} PaO2 records.")
    return result


def extract_fio2(cohort):
    print("\nExtracting FiO2...")

    cohort_map = cohort[
        ["stay_id", "intime", "outtime"]
    ]

    output = []
    counts = {
        "fraction": 0,
        "percentage": 0,
        "ambiguous": 0,
        "invalid": 0
    }

    columns = [
        "subject_id", "hadm_id", "stay_id",
        "itemid", "charttime", "valuenum"
    ]

    for chunk in pd.read_csv(
        CHART_FILE, compression="gzip",
        usecols=columns, chunksize=CHUNK_SIZE
    ):
        chunk = chunk[chunk["itemid"] == FIO2_ID]
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(
            chunk["charttime"], errors="coerce"
        )
        chunk["valuenum"] = pd.to_numeric(
            chunk["valuenum"], errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["stay_id", "charttime", "valuenum"]
        )

        chunk = chunk.merge(
            cohort_map, on="stay_id", how="inner"
        )

        chunk = chunk[
            (chunk["charttime"] >= chunk["intime"]) &
            (chunk["charttime"] <= chunk["outtime"])
        ].copy()

        if chunk.empty:
            continue

        value = chunk["valuenum"]

        fraction = (value > 0) & (value <= 1)
        percentage = (value >= 20) & (value <= 100)
        ambiguous = (value > 1) & (value < 20)

        counts["fraction"] += int(fraction.sum())
        counts["percentage"] += int(percentage.sum())
        counts["ambiguous"] += int(ambiguous.sum())
        counts["invalid"] += int(
            (~(fraction | percentage | ambiguous)).sum()
        )

        valid = fraction | percentage
        chunk = chunk[valid].copy()

        if chunk.empty:
            continue

        chunk["fio2_fraction"] = chunk["valuenum"].where(
            chunk["valuenum"] <= 1,
            chunk["valuenum"] / 100
        )

        chunk["fio2_source_value"] = chunk["valuenum"]
        chunk["fio2_source_format"] = (
            chunk["valuenum"].apply(
                lambda x: "fraction" if x <= 1 else "percentage"
            )
        )

        output.append(
            chunk[[
                "stay_id", "charttime", "fio2_source_value",
                "fio2_fraction", "fio2_source_format"
            ]]
        )

    result = pd.concat(output, ignore_index=True)
    result = result.drop_duplicates(
        subset=["stay_id", "charttime", "fio2_fraction"]
    )
    result.to_csv(FIO2_FILE, index=False)

    print(f"Saved {len(result):,} FiO2 records.")
    print("FiO2 classification:", counts)

    return result, counts


def pair_observations(pao2, fio2):
    print("\nPairing PaO2 with preceding FiO2...")

    left = pao2.sort_values(
        ["charttime", "stay_id"]
    ).copy()

    right = fio2.sort_values(
        ["charttime", "stay_id"]
    ).copy()

    paired = pd.merge_asof(
        left,
        right,
        on="charttime",
        by="stay_id",
        direction="backward",
        tolerance=PAIR_TOLERANCE
    )

    paired["pf_ratio"] = (
        paired["pao2_mmhg"] / paired["fio2_fraction"]
    )

    paired["fio2_match_flag"] = (
        paired["fio2_fraction"].notna()
    ).astype("int8")

    paired["pf_extreme_flag"] = (
        paired["pf_ratio"] > 600
    ).fillna(False).astype("int8")

    paired.to_csv(PAIRED_FILE, index=False)

    print(f"PaO2 records: {len(paired):,}")
    print(
        "Matched records:",
        int(paired["fio2_match_flag"].sum())
    )
    print(
        "Unmatched records:",
        int((paired["fio2_match_flag"] == 0).sum())
    )

    return paired


def main():
    cohort = load_cohort()

    pao2 = pd.read_csv(PAO2_FILE, parse_dates=["charttime"])
    fio2, fio2_counts = extract_fio2(cohort)

    paired = pair_observations(pao2, fio2)

    report = {
        "cohort_stays": int(cohort["stay_id"].nunique()),
        "pao2_records": int(len(pao2)),
        "fio2_records": int(len(fio2)),
        "paired_records": int(len(paired)),
        "matched_pao2_records": int(
            paired["fio2_match_flag"].sum()
        ),
        "unmatched_pao2_records": int(
            (paired["fio2_match_flag"] == 0).sum()
        ),
        "fio2_classification": fio2_counts,
        "pairing_tolerance_minutes": 60,
        "pao2_extreme_records": int(
            pao2["pao2_extreme_flag"].sum()
        )
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nRespiratory extraction complete.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
