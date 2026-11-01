from pathlib import Path

import numpy as np
import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.25):
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
    def __init__(self, d_model, max_len=32):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
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


class MIMICTemporalInference:
    def __init__(self, model_dir=None):
        self.device = torch.device("cpu")
        self.model_dir = (
            Path(model_dir) if model_dir
            else Path(__file__).resolve().parents[1] / "models" / "mimic"
        )

        self.models = {}
        self.checkpoints = {}

        self._load_model("bilstm")
        self._load_model("transformer")

    def _load_model(self, name):
        path = self.model_dir / f"mimic_{name}.pt"

        if not path.exists():
            raise FileNotFoundError(f"MIMIC checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        input_dim = checkpoint["model_config"]["input_dim"]

        if name == "bilstm":
            model = BiLSTMClassifier(input_dim=input_dim)
        elif name == "transformer":
            model = TransformerClassifier(input_dim=input_dim)
        else:
            raise ValueError(f"Unsupported model: {name}")

        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        self.models[name] = model
        self.checkpoints[name] = checkpoint

    def get_feature_columns(self):
        return self.checkpoints["bilstm"]["feature_columns"]

    def preprocess(self, X, name):
        checkpoint = self.checkpoints[name]
        prep = checkpoint["preprocessing"]

        medians = np.asarray(prep["medians"], dtype=np.float64)
        means = np.asarray(prep["means"], dtype=np.float64)
        scales = np.asarray(prep["scales"], dtype=np.float64)

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError("Expected input shape (batch, 6, 118).")

        if X.shape[1] != checkpoint["model_config"]["sequence_length"]:
            raise ValueError("Incorrect sequence length.")

        if X.shape[2] != len(checkpoint["feature_columns"]):
            raise ValueError("Feature count does not match checkpoint.")

        if len(medians) != X.shape[2]:
            raise ValueError("Median count does not match feature count.")

        if len(means) != X.shape[2] * 2 or len(scales) != X.shape[2] * 2:
            raise ValueError("Saved preprocessing dimensions are inconsistent.")

        missing = ~np.isfinite(X)
        filled = np.where(missing, medians.reshape(1, 1, -1), X)
        augmented = np.concatenate(
            [filled, missing.astype(np.float32)],
            axis=-1,
        )

        transformed = (
            (augmented - means.reshape(1, 1, -1))
            / scales.reshape(1, 1, -1)
        ).astype(np.float32)

        if not np.isfinite(transformed).all():
            raise ValueError("Non-finite values remain after preprocessing.")

        return torch.from_numpy(transformed).to(self.device)

    def predict(self, X, model_name="transformer"):
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        tensor = self.preprocess(X, model_name)

        with torch.no_grad():
            logits = self.models[model_name](tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        threshold = float(self.checkpoints[model_name]["threshold"])

        return {
            "model": model_name,
            "probabilities": probabilities.tolist(),
            "threshold": threshold,
            "predictions": (probabilities >= threshold).astype(int).tolist(),
        }
