from pathlib import Path

import numpy as np
import pandas as pd

from backend.mimic_inference import MIMICTemporalInference


class MIMICPatientService:
    def __init__(self, data_path=None):
        root = Path(__file__).resolve().parents[1]

        self.data_path = (
            Path(data_path)
            if data_path
            else root / "data" / "real" / "processed" / "splits" / "mimic_test.csv"
        )

        self.inference = MIMICTemporalInference()
        self.feature_columns = self.inference.get_feature_columns()

        self.records = None
        self.predictions = None

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MIMIC test data not found: {self.data_path}"
            )

        df = pd.read_csv(self.data_path)

        required = [
            "subject_id",
            "hadm_id",
            "stay_id",
            "assessment_time",
            "deterioration_label",
        ] + self.feature_columns

        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["assessment_time"] = pd.to_datetime(
            df["assessment_time"], errors="coerce"
        )

        if df["assessment_time"].isna().any():
            raise ValueError("Invalid assessment timestamps found.")

        if df.duplicated(["stay_id", "assessment_time"]).any():
            raise ValueError("Duplicate stay/timestamp records found.")

        patient_counts = df.groupby("stay_id")["subject_id"].nunique()
        if (patient_counts > 1).any():
            raise ValueError("An ICU stay maps to multiple patients.")

        self.records = df.sort_values(
            ["subject_id", "stay_id", "assessment_time"]
        ).reset_index(drop=True)

        return {
            "rows": int(len(self.records)),
            "patients": int(self.records["subject_id"].nunique()),
            "icu_stays": int(self.records["stay_id"].nunique()),
        }

    def run_inference(self):
        if self.records is None:
            self.load_data()

        sequences = []
        metadata = []

        for stay_id, group in self.records.groupby("stay_id", sort=False):
            group = group.sort_values("assessment_time").reset_index(drop=True)

            times = group["assessment_time"]
            features = group[self.feature_columns].to_numpy(
                dtype=np.float32
            )

            for end in range(5, len(group)):
                start = end - 5

                window_times = times.iloc[start:end + 1]
                gaps = window_times.diff().iloc[1:]

                if not (gaps.dt.total_seconds() == 3600).all():
                    continue

                sequences.append(features[start:end + 1])

                row = group.iloc[end]
                metadata.append({
                    "subject_id": int(row["subject_id"]),
                    "hadm_id": int(row["hadm_id"]),
                    "stay_id": int(stay_id),
                    "assessment_time": row["assessment_time"].isoformat(),
                    # Retained only for retrospective evaluation.
                    "outcome_label": (
                        int(row["deterioration_label"])
                        if pd.notna(row["deterioration_label"])
                        else None
                    ),
                })

        if not sequences:
            raise ValueError(
                "No valid six-hour consecutive sequences were found."
            )

        X = np.stack(sequences).astype(np.float32)

        bilstm = self.inference.predict(X, "bilstm")
        transformer = self.inference.predict(X, "transformer")

        results = []

        for i, item in enumerate(metadata):
            results.append({
                **item,
                "bilstm_probability": float(
                    bilstm["probabilities"][i]
                ),
                "bilstm_prediction": int(
                    bilstm["predictions"][i]
                ),
                "transformer_probability": float(
                    transformer["probabilities"][i]
                ),
                "transformer_prediction": int(
                    transformer["predictions"][i]
                ),
            })

        self.predictions = pd.DataFrame(results)

        return {
            "sequences_processed": int(len(self.predictions)),
            "patients": int(self.predictions["subject_id"].nunique()),
            "icu_stays": int(self.predictions["stay_id"].nunique()),
            "bilstm_threshold": bilstm["threshold"],
            "transformer_threshold": transformer["threshold"],
        }

    def get_roster(self):
        if self.predictions is None:
            self.run_inference()

        latest = (
            self.predictions
            .sort_values("assessment_time")
            .groupby("stay_id", as_index=False)
            .tail(1)
            .copy()
        )

        latest["risk_level"] = np.where(
            latest["transformer_prediction"] == 1,
            "High",
            "Below threshold",
        )

        return latest.drop(columns=["outcome_label"]).to_dict(
            orient="records"
        )

    def get_patient(self, stay_id):
        if self.predictions is None:
            self.run_inference()

        patient = self.predictions[
            self.predictions["stay_id"] == int(stay_id)
        ].sort_values("assessment_time")

        if patient.empty:
            return None

        return {
            "stay_id": int(stay_id),
            "subject_id": int(patient.iloc[0]["subject_id"]),
            "assessments": patient.drop(
                columns=["outcome_label"]
            ).to_dict(orient="records"),
        }
