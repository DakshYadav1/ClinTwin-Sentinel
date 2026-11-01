
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from train_mimic_temporal_models import (
    BiLSTMClassifier,
    TransformerClassifier,
    SEQ_DIR,
    MODEL_DIR,
)

DEVICE = torch.device("cpu")
REPORT_DIR = Path("data/real/processed")


def transform(X, prep):
    medians = np.asarray(prep["medians"])
    means = np.asarray(prep["means"])
    scales = np.asarray(prep["scales"])

    missing = ~np.isfinite(X)
    filled = np.where(missing, medians.reshape(1, 1, -1), X)

    augmented = np.concatenate(
        [filled, missing.astype(np.float32)], axis=-1
    )

    return (
        (augmented - means.reshape(1, 1, -1))
        / scales.reshape(1, 1, -1)
    ).astype(np.float32)


def calculate_metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("Loading held-out TEST sequences...")

    data = np.load(
        SEQ_DIR / "mimic_test_sequences.npz",
        allow_pickle=False,
    )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)

    print(f"Test sequences: {len(y):,}")
    print(f"Positive cases: {int(y.sum()):,}")
    print(f"Negative cases: {int((y == 0).sum()):,}")

    results = {
        "evaluation": "one-time held-out test evaluation",
        "test_sequences": int(len(y)),
        "positive_cases": int(y.sum()),
        "negative_cases": int((y == 0).sum()),
        "thresholds_source": "locked using validation data",
        "models": {},
    }

    for name in ["bilstm", "transformer"]:
        print(f"\nEvaluating {name.upper()}...")

        artifact = torch.load(
            MODEL_DIR / f"mimic_{name}.pt",
            map_location=DEVICE,
            weights_only=False,
        )

        X_scaled = transform(X, artifact["preprocessing"])

        input_dim = artifact["model_config"]["input_dim"]

        if name == "bilstm":
            model = BiLSTMClassifier(input_dim=input_dim)
        else:
            model = TransformerClassifier(input_dim=input_dim)

        model.load_state_dict(artifact["state_dict"])
        model.to(DEVICE)
        model.eval()

        probabilities = []

        with torch.no_grad():
            for start in range(0, len(X_scaled), 128):
                batch = torch.from_numpy(
                    X_scaled[start:start + 128]
                ).to(DEVICE)

                output = torch.sigmoid(model(batch))
                probabilities.extend(output.cpu().numpy())

        probabilities = np.asarray(probabilities)

        threshold = float(artifact["threshold"])

        metrics = calculate_metrics(
            y, probabilities, threshold
        )

        results["models"][name] = metrics

        for key, value in metrics.items():
            print(f"{key}: {value}")

    output_path = REPORT_DIR / "mimic_temporal_test_report.json"

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nTest report saved: {output_path}")
    print("Evaluation complete. Thresholds were not changed.")


if __name__ == "__main__":
    main()
