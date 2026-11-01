"""Train leakage-safe baseline MIMIC deterioration models.

Uses the existing patient-level train/validation splits only.
The held-out test set is deliberately NOT loaded or evaluated here.

Models:
  1. Median-imputed, standardized, class-weighted logistic regression
  2. Median-imputed, class-weighted random forest

Preprocessing is fitted separately inside each training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data" / "real" / "processed" / "splits"
MODEL_DIR = ROOT / "models" / "mimic"
REPORT_DIR = ROOT / "data" / "real" / "processed"

TARGET = "deterioration_label"
EXCLUDED = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "assessment_time",
    TARGET,
}


def metrics_at_threshold(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    train_path = SPLIT_DIR / "mimic_train.csv"
    validation_path = SPLIT_DIR / "mimic_validation.csv"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run create_mimic_patient_splits.py first.")

    print("Loading train and validation splits (test remains untouched)...")
    train = pd.read_csv(train_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)

    feature_columns = [c for c in train.columns if c not in EXCLUDED]
    if not feature_columns:
        raise ValueError("No predictor columns found.")

    non_numeric = train[feature_columns].select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric predictors require explicit handling: {non_numeric}")

    missing_validation = set(feature_columns) - set(validation.columns)
    if missing_validation:
        raise ValueError(f"Validation split missing predictors: {sorted(missing_validation)}")

    X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET].astype(int)
    X_val = validation[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_val = validation[TARGET].astype(int)

    if set(y_train.unique()) != {0, 1} or set(y_val.unique()) != {0, 1}:
        raise ValueError("Both train and validation must contain both target classes.")

    print(f"Training rows: {len(train):,} | Validation rows: {len(validation):,}")
    print(f"Predictors: {len(feature_columns)}")
    print("Excluded from predictors:", sorted(EXCLUDED))

    # Median imputation and missingness indicators are fitted on training only.
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models = {
        "logistic_regression": logistic,
        "random_forest": forest,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "task": "predict >=2 point SOFA increase at any of +2h to +6h",
        "cohort": "strict five-future-assessment labelled cohort",
        "split_unit": "subject_id",
        "predictor_count": len(feature_columns),
        "excluded_columns": sorted(EXCLUDED),
        "preprocessing": (
            "Median imputation; missingness indicators; scaling for logistic "
            "regression; all preprocessing fitted on training split only."
        ),
        "test_set_used": False,
        "models": {},
    }

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        model_metrics = metrics_at_threshold(y_val, probabilities, threshold=0.5)
        results["models"][name] = model_metrics

        artifact_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_columns": feature_columns,
                "target": TARGET,
                "threshold": 0.5,
                "split_unit": "subject_id",
            },
            artifact_path,
        )
        print(json.dumps(model_metrics, indent=2))
        print(f"Saved model: {artifact_path}")

    report_path = REPORT_DIR / "mimic_baseline_validation_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nValidation report saved: {report_path}")
    print("The test split was not loaded or evaluated.")


if __name__ == "__main__":
    main()
