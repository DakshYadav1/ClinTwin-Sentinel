"""Train MIMIC temporal models: BiLSTM and Transformer only.

Uses:
  data/real/processed/sequences/mimic_train_sequences.npz
  data/real/processed/sequences/mimic_validation_sequences.npz

The test sequence file is intentionally not loaded.
Imputation and standardization parameters are fitted on training timesteps only.
Missingness indicators are appended to each timestep.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "data" / "real" / "processed" / "sequences"
MODEL_DIR = ROOT / "models" / "mimic"
REPORT_DIR = ROOT / "data" / "real" / "processed"

SEED = 42
BATCH_SIZE = 128
MAX_EPOCHS = 40
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
DEVICE = torch.device("cpu")  # Current environment has CPU-only PyTorch.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz(name: str):
    path = SEQ_DIR / f"mimic_{name}_sequences.npz"

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)

    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        data["subject_id"],
        data["stay_id"],
        None,  # Timestamps are not needed for training
        data["feature_columns"].astype(str).tolist(),
    )


def fit_preprocessor(X_train: np.ndarray):
    # Fit medians using training data only, across training sequences/timesteps.
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    medians = np.zeros(flat.shape[1], dtype=np.float64)
    all_missing = np.zeros(flat.shape[1], dtype=bool)

    for j in range(flat.shape[1]):
        finite = flat[np.isfinite(flat[:, j]), j]
        if finite.size:
            medians[j] = np.median(finite)
        else:
            medians[j] = 0.0
            all_missing[j] = True

    def impute_and_append(X):
        missing = ~np.isfinite(X)
        filled = np.where(missing, medians.reshape(1, 1, -1), X)
        # Append missingness indicators so missing is not mistaken for normal.
        return np.concatenate([filled, missing.astype(np.float32)], axis=-1)

    train_aug = impute_and_append(X_train).astype(np.float64)
    means = train_aug.reshape(-1, train_aug.shape[-1]).mean(axis=0)
    scales = train_aug.reshape(-1, train_aug.shape[-1]).std(axis=0)
    scales[scales < 1e-8] = 1.0

    return medians, means, scales, all_missing, impute_and_append


def transform(X, medians, means, scales):
    missing = ~np.isfinite(X)
    filled = np.where(missing, medians.reshape(1, 1, -1), X)
    augmented = np.concatenate([filled, missing.astype(np.float32)], axis=-1)
    return ((augmented - means.reshape(1, 1, -1)) /
            scales.reshape(1, 1, -1)).astype(np.float32)


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.position(x)
        x = self.encoder(x)
        return self.head(self.norm(x[:, -1, :])).squeeze(-1)


def evaluate(model, loader):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p.append(probs)
            all_y.append(yb.numpy())
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    pred = (p >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": THRESHOLD,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_one(name, model, train_loader, val_loader, y_train, feature_columns, preproc):
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    pos_weight = torch.tensor([negatives / max(positives, 1)], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        val_metrics = evaluate(model, val_loader)
        epoch_loss = running_loss / max(seen, 1)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "validation_auprc": val_metrics["auprc"],
            "validation_auroc": val_metrics["auroc"],
            "validation_brier": val_metrics["brier_score"],
        })
        print(
            f"{name} | epoch {epoch:02d} | loss={epoch_loss:.4f} "
            f"| val_AUPRC={val_metrics['auprc']:.4f} "
            f"| val_AUROC={val_metrics['auroc']:.4f}"
        )

        # AUPRC is the early-stopping/model-selection metric for imbalance.
        if val_metrics["auprc"] > best_ap + 1e-5:
            best_ap = val_metrics["auprc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint captured for {name}")

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader)

    artifact = {
        "model_name": name,
        "state_dict": model.state_dict(),
        "model_config": {
            "input_dim": len(feature_columns) * 2,
            "sequence_length": 6,
        },
        "feature_columns": feature_columns,
        "preprocessing": {
            "medians": preproc[0].tolist(),
            "means": preproc[1].tolist(),
            "scales": preproc[2].tolist(),
            "all_missing_training_features": [
                feature_columns[i] for i, flag in enumerate(preproc[3]) if flag
            ],
            "missingness_indicators": True,
        },
        "threshold": THRESHOLD,
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"mimic_{name}.pt"
    torch.save(artifact, model_path)

    return {
        "best_epoch": best_epoch,
        "validation_metrics": final_metrics,
        "history": history,
        "model_path": str(model_path),
    }


def main():
    set_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print(f"Device: {DEVICE}")
    print("Loading training and validation sequences only; test remains untouched.")

    X_train, y_train, train_subjects, _, _, feature_columns = load_npz("train")
    X_val, y_val, val_subjects, _, _, val_features = load_npz("validation")

    if feature_columns != val_features:
        raise ValueError("Train/validation feature order mismatch.")
    if set(np.unique(train_subjects)) & set(np.unique(val_subjects)):
        raise ValueError("Patient overlap between train and validation sequences.")
    if X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Train/validation sequence shape mismatch.")

    print(f"Train: {X_train.shape}, positives={int(y_train.sum()):,}")
    print(f"Validation: {X_val.shape}, positives={int(y_val.sum()):,}")
    print(f"Features per timestep: {len(feature_columns)}")

    medians, means, scales, all_missing, _ = fit_preprocessor(X_train)
    X_train_scaled = transform(X_train, medians, means, scales)
    X_val_scaled = transform(X_val, medians, means, scales)

    # Ensure preprocessing did not produce non-finite values.
    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_val_scaled).all():
        raise ValueError("Non-finite values remain after preprocessing.")

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_scaled),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    preproc = (medians, means, scales, all_missing)
    input_dim = X_train_scaled.shape[-1]

    models = {
        "bilstm": BiLSTMClassifier(input_dim=input_dim),
        "transformer": TransformerClassifier(input_dim=input_dim),
    }

    report = {
        "task": "predict deterioration_label at final sequence timestamp",
        "sequence_length_hours": 6,
        "features_per_timestep": len(feature_columns),
        "model_input_features_including_missingness_indicators": int(input_dim),
        "device": str(DEVICE),
        "seed": SEED,
        "test_set_loaded": False,
        "preprocessing_fit_on": "training sequences only",
        "early_stopping_metric": "validation AUPRC",
        "threshold": THRESHOLD,
        "models": {},
    }

    for name, model in models.items():
        print(f"\n===== Training {name.upper()} =====")
        model = model.to(DEVICE)
        report["models"][name] = train_one(
            name, model, train_loader, val_loader, y_train,
            feature_columns, preproc
        )

    report_path = REPORT_DIR / "mimic_temporal_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nValidation report saved: {report_path}")
    print("Saved BiLSTM and Transformer checkpoints under models/mimic/.")
    print("Test sequences were not loaded or evaluated.")


if __name__ == "__main__":
    main()
