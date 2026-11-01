import pandas as pd

cohort = pd.read_csv("data/real/processed/strict_icu_cohort.csv")

stay_map = cohort[
    ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
].copy()

stay_map["intime"] = pd.to_datetime(stay_map["intime"])
stay_map["outtime"] = pd.to_datetime(stay_map["outtime"])

sources = [
    ("VITAL SIGNS", "data/real/processed/cleaned_vitals.csv", "stay"),
    ("LABORATORY RESULTS", "data/real/processed/cleaned_labevents.csv", "admission"),
    ("URINE OUTPUT", "data/real/processed/cleaned_urine_output.csv", "stay"),
]

for name, path, match_type in sources:
    print(f"\n========== {name} ==========")

    total = linked = inside_count = outside_count = 0
    matched_stays = set()
    feature_counts = {}

    for chunk in pd.read_csv(path, chunksize=500_000):
        total += len(chunk)

        if match_type == "stay":
            chunk["stay_id"] = pd.to_numeric(
                chunk["stay_id"], errors="coerce"
            )
            chunk = chunk.dropna(subset=["stay_id"]).copy()
            chunk["stay_id"] = chunk["stay_id"].astype("int64")

            merged = chunk.merge(
                stay_map[["stay_id", "intime", "outtime"]],
                on="stay_id",
                how="inner"
            )
        else:
            chunk["subject_id"] = pd.to_numeric(
                chunk["subject_id"], errors="coerce"
            )
            chunk["hadm_id"] = pd.to_numeric(
                chunk["hadm_id"], errors="coerce"
            )

            chunk = chunk.dropna(
                subset=["subject_id", "hadm_id"]
            ).copy()

            chunk["subject_id"] = chunk["subject_id"].astype("int64")
            chunk["hadm_id"] = chunk["hadm_id"].astype("int64")

            merged = chunk.merge(
                stay_map,
                on=["subject_id", "hadm_id"],
                how="inner"
            )

        linked += len(merged)

        if merged.empty:
            continue

        times = pd.to_datetime(merged["charttime"], errors="coerce")

        valid = times.notna()
        inside = (
            valid
            & (times >= merged["intime"])
            & (times <= merged["outtime"])
        )

        inside_count += int(inside.sum())
        outside_count += int((valid & ~inside).sum())

        matched_stays.update(
            merged.loc[inside, "stay_id"].unique()
        )

        if "feature" in merged.columns:
            counts = merged.loc[inside, "feature"].value_counts()
            for key, value in counts.items():
                feature_counts[key] = (
                    feature_counts.get(key, 0) + int(value)
                )
        elif "itemid" in merged.columns:
            counts = merged.loc[inside, "itemid"].value_counts()
            for key, value in counts.items():
                feature_counts[str(key)] = (
                    feature_counts.get(str(key), 0) + int(value)
                )

    print("Source rows:", f"{total:,}")
    print("Rows linked to cohort:", f"{linked:,}")
    print("Rows inside ICU time window:", f"{inside_count:,}")
    print("Rows outside ICU time window:", f"{outside_count:,}")
    print("Unique ICU stays with records:", f"{len(matched_stays):,}")
    print("ICU stays without records:", f"{len(cohort) - len(matched_stays):,}")
    print("Feature/item counts inside ICU windows:")

    for key, value in sorted(feature_counts.items()):
        print(f"  {key}: {value:,}")

print("\nCoverage inspection complete.")
