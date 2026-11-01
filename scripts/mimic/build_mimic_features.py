from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data" / "real" / "processed"
LABELS = P / "mimic_deterioration_dataset.csv"
COHORT = P / "strict_icu_cohort.csv"
OUT = P / "mimic_features.csv"
REPORT = P / "mimic_feature_report.json"
CHUNK = 1_000_000

LAB_MAP = {
    50813: "lactate", 53154: "lactate",
    50885: "bilirubin", 53089: "bilirubin",
    50912: "creatinine", 52546: "creatinine",
    51265: "platelets", 53189: "platelets",
    51300: "wbc", 51301: "wbc", 51755: "wbc", 51756: "wbc",
}
VITALS = [
    "heart_rate", "respiratory_rate", "temperature",
    "arterial_systolic_bp", "arterial_diastolic_bp", "arterial_mean_bp",
    "nibp_systolic_bp", "nibp_diastolic_bp", "nibp_mean_bp",
]
SOFA = [
    "cardiovascular_sofa", "respiratory_sofa", "coagulation_sofa",
    "liver_sofa", "renal_sofa", "neurological_sofa", "total_sofa",
]


def anchors_load():
    d = pd.read_csv(LABELS, parse_dates=["assessment_time"])
    d = d[d.deterioration_label.notna()].copy()
    d["deterioration_label"] = d.deterioration_label.astype("int8")
    d = d.merge(
        pd.read_csv(COHORT, usecols=["stay_id", "intime", "outtime"]),
        on="stay_id", how="left", validate="many_to_one"
    )
    d["intime"] = pd.to_datetime(d.intime)
    d["outtime"] = pd.to_datetime(d.outtime)
    if d[["intime", "outtime"]].isna().any().any():
        raise ValueError("Some labelled stays could not be linked to strict_icu_cohort.")
    d = d.sort_values(["assessment_time", "stay_id"]).reset_index(drop=True)
    if d.duplicated(["stay_id", "assessment_time"]).any():
        raise ValueError("Duplicate stay/time prediction anchors.")
    return d


def window_stats(anchors, measurements, feature):
    """Latest, age, 6h/24h mean and count; only prior/current observations."""
    out = pd.DataFrame(index=anchors.index)
    names = [
        f"{feature}__latest", f"{feature}__age_hours",
        f"{feature}__mean_6h", f"{feature}__count_6h",
        f"{feature}__mean_24h", f"{feature}__count_24h",
    ]
    for n in names:
        out[n] = np.nan if "count" not in n else 0.0

    m = measurements.loc[measurements.feature == feature,
                         ["stay_id", "charttime", "value"]].copy()
    if m.empty:
        return out
    m["charttime"] = pd.to_datetime(m.charttime)
    m["value"] = pd.to_numeric(m.value, errors="coerce")
    m = m.dropna().sort_values(["charttime", "stay_id"])
    ag = anchors[["stay_id", "assessment_time"]].copy()
    ag["_idx"] = anchors.index
    ag = ag.sort_values(["assessment_time", "stay_id"])
    mg = m.sort_values(["charttime", "stay_id"])

    # Latest observation at or before assessment, no older than 24h.
    latest = pd.merge_asof(
        ag, mg.rename(columns={"charttime": "measurement_time", "value": "latest"}),
        left_on="assessment_time", right_on="measurement_time",
        by="stay_id", direction="backward", tolerance=pd.Timedelta(hours=24)
    ).set_index("_idx")
    out.loc[latest.index, f"{feature}__latest"] = latest.latest
    out.loc[latest.index, f"{feature}__age_hours"] = (
        latest.assessment_time - latest.measurement_time
    ).dt.total_seconds() / 3600

    # Efficient per-stay prefix sums for rolling means and counts.
    groups = {sid: g for sid, g in m.groupby("stay_id", sort=False)}
    for sid, idx in anchors.groupby("stay_id", sort=False).indices.items():
        g = groups.get(sid)
        if g is None:
            continue
        t = g.charttime.to_numpy(dtype="datetime64[ns]")
        v = g.value.to_numpy(dtype="float64")
        at = anchors.loc[idx, "assessment_time"].to_numpy(dtype="datetime64[ns]")
        right = np.searchsorted(t, at, side="right")
        prefix = np.r_[0.0, np.cumsum(v)]
        for h in (6, 24):
            left = np.searchsorted(t, at - np.timedelta64(h, "h"), side="left")
            count = right - left
            total = prefix[right] - prefix[left]
            mean = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
            out.loc[np.asarray(idx), f"{feature}__mean_{h}h"] = mean
            out.loc[np.asarray(idx), f"{feature}__count_{h}h"] = count
    return out


def collect_vitals(anchors):
    print("Reading cleaned vitals in chunks...")
    stays = set(anchors.stay_id.unique())
    parts = []
    cols = ["stay_id", "charttime", "feature", "value"]
    for c in pd.read_csv(P / "cleaned_vitals.csv", usecols=cols,
                         chunksize=CHUNK, low_memory=False):
        c = c[c.stay_id.isin(stays) & c.feature.isin(VITALS)]
        if len(c):
            parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def collect_labs(anchors):
    print("Reading cleaned labs in chunks and linking to ICU stay times...")
    stays = anchors[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]].drop_duplicates()
    parts = []
    cols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    for c in pd.read_csv(P / "cleaned_labevents.csv", usecols=cols,
                         chunksize=CHUNK, low_memory=False):
        c = c[c.itemid.isin(LAB_MAP)]
        if c.empty:
            continue
        c = c.merge(stays, on=["subject_id", "hadm_id"], how="inner")
        c["charttime"] = pd.to_datetime(c.charttime)
        c = c[(c.charttime >= c.intime) & (c.charttime <= c.outtime)]
        if c.empty:
            continue
        c["feature"] = c.itemid.map(LAB_MAP)
        c["value"] = pd.to_numeric(c.valuenum, errors="coerce")
        parts.append(c[["stay_id", "charttime", "feature", "value"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["stay_id", "charttime", "feature", "value"]
    )


def urine_features(anchors):
    print("Reading urine output...")
    stays = set(anchors.stay_id.unique())
    parts = []
    cols = ["stay_id", "charttime", "value", "irrigation_overlap_flag"]
    for c in pd.read_csv(P / "cleaned_urine_output.csv", usecols=cols,
                         chunksize=CHUNK, low_memory=False):
        c = c[c.stay_id.isin(stays) & (c.irrigation_overlap_flag == 0)].copy()
        c["value"] = pd.to_numeric(c.value, errors="coerce")
        c = c[c.value.between(0, 10000)]
        if len(c):
            c["charttime"] = pd.to_datetime(c.charttime)
            parts.append(c[["stay_id", "charttime", "value"]])
    u = pd.concat(parts, ignore_index=True)
    u = u.groupby(["stay_id", "charttime"], as_index=False).value.sum()
    out = pd.DataFrame(index=anchors.index)
    for h in (6, 24):
        out[f"urine_output__sum_{h}h"] = np.nan
    groups = {sid: g.sort_values("charttime") for sid, g in u.groupby("stay_id")}
    for sid, idx in anchors.groupby("stay_id", sort=False).indices.items():
        g = groups.get(sid)
        if g is None:
            continue
        t = g.charttime.to_numpy(dtype="datetime64[ns]")
        v = g.value.to_numpy(dtype="float64")
        at = anchors.loc[idx, "assessment_time"].to_numpy(dtype="datetime64[ns]")
        right = np.searchsorted(t, at, side="right")
        prefix = np.r_[0.0, np.cumsum(v)]
        for h in (6, 24):
            left = np.searchsorted(t, at - np.timedelta64(h, "h"), side="left")
            out.loc[np.asarray(idx), f"urine_output__sum_{h}h"] = prefix[right] - prefix[left]
    return out


def respiratory_features(anchors):
    print("Adding PaO2/FiO2 and ventilation...")
    p = pd.read_csv(
        P / "paired_pao2_fio2.csv",
        usecols=["stay_id", "charttime", "pao2_mmhg", "fio2_fraction", "pf_ratio"],
        parse_dates=["charttime"]
    ).sort_values(["charttime", "stay_id"])
    a = anchors[["stay_id", "assessment_time"]].copy()
    a["_idx"] = anchors.index
    a = a.sort_values(["assessment_time", "stay_id"])
    z = pd.merge_asof(
        a, p.rename(columns={"charttime": "measurement_time"}),
        left_on="assessment_time", right_on="measurement_time",
        by="stay_id", direction="backward", tolerance=pd.Timedelta(hours=6)
    ).set_index("_idx")
    out = pd.DataFrame(index=anchors.index)
    for col, name in [
        ("pao2_mmhg", "pao2__latest"),
        ("fio2_fraction", "fio2__latest"),
        ("pf_ratio", "pf_ratio__latest"),
    ]:
        out[name] = z[col].reindex(anchors.index)
    out["pf_ratio__age_hours"] = (
        anchors.assessment_time - z.measurement_time.reindex(anchors.index)
    ).dt.total_seconds() / 3600

    v = pd.read_csv(
        P / "hourly_ventilation_support.csv",
        usecols=["stay_id", "assessment_time", "invasive_ventilation",
                 "noninvasive_ventilation", "ventilation_support",
                 "ventilation_overlap_flag", "ventilation_status_review_flag"],
        parse_dates=["assessment_time"]
    )
    out = out.join(
        anchors[["stay_id", "assessment_time"]].merge(
            v, on=["stay_id", "assessment_time"], how="left",
            validate="one_to_one"
        )[[
            "invasive_ventilation", "noninvasive_ventilation",
            "ventilation_support", "ventilation_overlap_flag",
            "ventilation_status_review_flag"
        ]].set_axis(anchors.index)
    )
    return out


def main():
    anchors = anchors_load()
    print(f"Labelled anchors: {len(anchors):,}; patients: {anchors.subject_id.nunique():,}; stays: {anchors.stay_id.nunique():,}")

    parts = []
    vitals = collect_vitals(anchors)
    for i, f in enumerate(VITALS, 1):
        print(f"Vital {i}/{len(VITALS)}: {f}")
        parts.append(window_stats(anchors, vitals, f))

    labs = collect_labs(anchors)
    for i, f in enumerate(sorted(set(LAB_MAP.values())), 1):
        print(f"Lab {i}/{len(set(LAB_MAP.values()))}: {f}")
        parts.append(window_stats(anchors, labs, f))

    parts.append(urine_features(anchors))
    parts.append(respiratory_features(anchors))

    # Keep the baseline SOFA components as predictors. Future-derived columns
    # are intentionally not loaded into this feature matrix.
    base = anchors[[
        "subject_id", "hadm_id", "stay_id", "assessment_time",
        *SOFA, "deterioration_label"
    ]].copy()
    features = pd.concat(parts, axis=1)
    result = pd.concat([base, features], axis=1)

    result.to_csv(OUT, index=False)
    report = {
        "rows": int(len(result)),
        "patients": int(result.subject_id.nunique()),
        "stays": int(result.stay_id.nunique()),
        "positive_labels": int(result.deterioration_label.sum()),
        "negative_labels": int((result.deterioration_label == 0).sum()),
        "feature_columns": [c for c in result.columns if c not in base.columns],
        "missing_fraction": {
            c: float(result[c].isna().mean())
            for c in result.columns
            if c not in {
                "subject_id", "hadm_id", "stay_id",
                "assessment_time", "deterioration_label"
            }
        },
        "temporal_rule": "Only measurements with charttime <= assessment_time; latest measurements limited to 24h (6h for PaO2/FiO2).",
        "warning": "Research feature matrix; review missingness and label ascertainment before training. Urine sums are raw recorded volume sums, not weight-normalised mL/kg/h.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFeature matrix saved:", OUT)
    print("Rows:", len(result), "| columns:", len(result.columns))
    print("Report saved:", REPORT)


if __name__ == "__main__":
    main()
