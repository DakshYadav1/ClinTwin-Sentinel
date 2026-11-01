import os
import pandas as pd

paths = {
    "chartevents": "data/real/raw/mimiciv-3.1/icu/chartevents.csv.gz",
    "outputevents": "data/real/raw/mimiciv-3.1/icu/outputevents.csv.gz",
    "inputevents": "data/real/raw/mimiciv-3.1/icu/inputevents.csv.gz",
    "icu_dictionary": "data/real/raw/icu/d_items.csv.gz",
    "lab_dictionary": "data/real/raw/hosp/d_labitems.csv.gz",
    "labevents": "data/real/raw/mimiciv-3.1/hosp/labevents.csv.gz",
}

print("\n========== FILE AVAILABILITY ==========")
for name, path in paths.items():
    print(f"{name}: {'AVAILABLE' if os.path.exists(path) else 'NOT DOWNLOADED'}")

icu_dict = pd.read_csv(paths["icu_dictionary"])
lab_dict = pd.read_csv(paths["lab_dictionary"])

icu_terms = [
    "glasgow", "gcs", "fio2", "pao2", "oxygen saturation",
    "norepinephrine", "noradrenaline", "vasopressin",
    "dopamine", "dobutamine", "epinephrine", "adrenaline",
    "phenylephrine", "mean blood pressure"
]

print("\n========== ICU DICTIONARY ==========")
for term in icu_terms:
    matches = icu_dict[
        icu_dict["label"].str.contains(term, case=False, na=False)
    ]
    if not matches.empty:
        print(f"\n--- {term.upper()} ---")
        cols = [
            c for c in
            ["itemid", "label", "category", "unitname", "linksto"]
            if c in matches.columns
        ]
        print(matches[cols].head(25).to_string(index=False))

lab_terms = [
    "pO2", "oxygen", "bilirubin", "platelet",
    "creatinine", "lactate"
]

print("\n========== LAB DICTIONARY ==========")
for term in lab_terms:
    matches = lab_dict[
        lab_dict["label"].str.contains(term, case=False, na=False)
    ]
    if not matches.empty:
        print(f"\n--- {term.upper()} ---")
        cols = [
            c for c in ["itemid", "label", "fluid", "category"]
            if c in matches.columns
        ]
        print(matches[cols].head(25).to_string(index=False))

print("\nInspection complete.")
