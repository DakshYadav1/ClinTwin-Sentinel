import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from train_mimic_temporal_models import (
    BiLSTMClassifier,
    TransformerClassifier,
    SEQ_DIR,
    MODEL_DIR,
    REPORT_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cpu")


def transform(X, prep):
    medians = np.array(prep["medians"])
    means = np.array(prep["means"])
    scales = np.array(prep["scales"])

    missing = ~np.isfinite(X)
    filled = np.where(missing, medians.reshape(1, 1, -1), X)
    augmented = np.concatenate(
        [filled, missing.astype(np.float32)], axis=-1
    )

    return (
        (augmented - means.reshape(1, 1, -1))
        / scales.reshape(1, 1, -1)
    ).astype(np.float32)


def get_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "threshold": round(float(threshold), 2),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    data = np.load(
        SEQ_DIR / "mimic_validation_sequences.npz",
        allow_pickle=False,
    )
    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)

    results = {
        "evaluation": "validation threshold analysis",
        "test_set_loaded": False,
        "models": {},
    }

    for name in ["bilstm", "transformer"]:
        path = MODEL_DIR / f"mimic_{name}.pt"

        artifact = torch.load(
            path, map_location=DEVICE, weights_only=False
        )

        X_scaled = transform(X, artifact["preprocessing"])

        if name == "bilstm":
            model = BiLSTMClassifier(
                input_dim=artifact["model_config"]["input_dim"]
            )
        else:
            model = TransformerClassifier(
                input_dim=artifact["model_config"]["input_dim"]
            )

        model.load_state_dict(artifact["state_dict"])
        model.to(DEVICE)
        model.eval()

        predictions = []

        with torch.no_grad():
            for start in range(0, len(X_scaled), 128):
                batch = torch.from_numpy(
                    X_scaled[start:start + 128]
                ).to(DEVICE)

                probabilities = torch.sigmoid(model(batch))
                predictions.extend(probabilities.cpu().numpy())

        p = np.asarray(predictions)

        thresholds = [
            get_metrics(y, p, t)
            for t in np.arange(0.10, 0.91, 0.05)
        ]

        results["models"][name] = {
            "thresholds": thresholds,
            "best_f1_threshold": max(
                thresholds, key=lambda x: x["f1"]
            ),
            "threshold_for_recall_80": next(
                (
                    row for row in thresholds
                    if row["recall"] >= 0.80
                ),
                None,
            ),
        }

        print(f"\n===== {name.upper()} =====")
        for row in thresholds:
            print(
                f"Threshold={row['threshold']:.2f} | "
                f"Precision={row['precision']:.3f} | "
                f"Recall={row['recall']:.3f} | "
                f"Specificity={row['specificity']:.3f} | "
                f"F1={row['f1']:.3f}"
            )

        print("Best F1:", results["models"][name]["best_f1_threshold"])
        print("Recall >= 0.80:", results["models"][name]["threshold_for_recall_80"])

    output = REPORT_DIR / "mimic_threshold_analysis.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nReport saved: {output}")
    print("Test sequences were not loaded.")


if __name__ == "__main__":
    main()