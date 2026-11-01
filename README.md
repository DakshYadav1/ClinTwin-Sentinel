# ClinTwin Sentinel
### ICU Digital Twin Research Prototype | BioGears + MIMIC-IV + Deep Learning

ClinTwin Sentinel is an ICU Digital Twin research prototype that combines physiological simulation, retrospective ICU data analysis, and temporal deep-learning models to estimate patient deterioration risk.

The system integrates two independent pathways:

- **BioGears:** Physiology-based simulation, machine-learning inference, and What-If intervention analysis.
- **MIMIC-IV:** Retrospective ICU data processing and independent BiLSTM and Transformer deterioration prediction.

The application uses a FastAPI backend and React dashboard to visualize model outputs and patient trajectories.

> **Research prototype only.** ClinTwin Sentinel is not a medical device and must not be used for diagnosis, treatment decisions, or real-time clinical care.

---

## 1. Key Features

### BioGears Physiological Simulation
- Physiological data generation using BioGears.
- Five-minute resampling and 46-feature engineering.
- Temporal sequence modelling using BiLSTM.
- Deterioration-risk estimation and 256-dimensional latent-state representation.
- FastAPI integration and interactive React visualization.

### What-If Intervention Simulation
- Simulates hypothetical physiological interventions.
- Demonstrates a 100 mL saline bolus scenario.
- Supports visualization and comparison of simulated physiological responses.
- Provides a framework for exploring intervention scenarios.

### MIMIC-IV Retrospective Analysis
- Processes retrospective ICU data from MIMIC-IV v3.1.
- Extracts vital signs, laboratory measurements, urine output, and vasopressor information.
- Engineers hourly physiological and SOFA-related features.
- Constructs six-hour temporal sequences.
- Predicts deterioration using two independent deep-learning models.

### Machine Learning Models

| Model | Architecture | Purpose |
|---|---|---|
| BioGears BiLSTM | Bidirectional LSTM | Temporal physiological modelling |
| MIMIC-IV BiLSTM | Two-layer BiLSTM | Retrospective deterioration prediction |
| MIMIC-IV Transformer | Transformer Encoder | Attention-based temporal prediction |

### Dashboard
- Separate BioGears and MIMIC-IV source modes.
- Patient and ICU stay selection.
- Deterioration-risk visualization.
- Independent model predictions.
- Prediction trajectory visualization.
- Vitals and feature attribution panels.
- Latent-state and attention visualizations.
- What-If simulation interface.

---

## 2. System Architecture

### BioGears Pipeline

```text
BioGears Simulation
        |
Physiological CSV
        |
5-Minute Resampling
        |
46 Feature Engineering
        |
72 x 46 Input Tensor
        |
BioGears BiLSTM
        |
Risk Score + 256-D Latent State
        |
FastAPI Backend
        |
React Dashboard / What-If
```

### MIMIC-IV Pipeline

```text
MIMIC-IV v3.1
      |
Data Extraction and Cleaning
      |
Hourly Feature Engineering
      |
SOFA Calculation and Labelling
      |
Patient-Level Train / Validation / Test Split
      |
Six-Hour Sequence Construction
      |
BiLSTM             Transformer
    |                    |
    +---------+----------+
              |
      Independent Predictions
              |
         FastAPI
              |
       React Dashboard
```

The two pathways operate independently and use separate inference workflows.

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python, JavaScript |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS |
| Machine Learning | PyTorch, BiLSTM, Transformer |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Recharts, Framer Motion |
| Physiological Simulation | BioGears |
| Dataset | MIMIC-IV v3.1 |
| Version Control | Git, GitHub |

---

## 4. Repository Structure

```text
ClinTwin-Sentinel/
│
├── backend/
│   ├── api.py
│   ├── mimic_inference.py
│   ├── mimic_service.py
│   ├── inference_service.py
│   ├── biogears_service.py
│   ├── biogears_inference.py
│   ├── lstm_model.py
│   ├── transformer_model.py
│   ├── mimic_adapter/
│   └── synthetic_data/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── App.jsx
│   ├── package.json
│   └── package-lock.json
│
├── models/
│   ├── biogears_lstm_model.pt
│   ├── biogears_transformer_model.pt
│   └── biogears_scaler.joblib
│
├── scripts/
│   └── mimic/
│
├── data/
│   └── real/
│
├── docs/
│   ├── BIOGEARS_IMPLEMENTATION.md
│   ├── MIMIC_IV_SETUP.md
│   ├── RUN_EVERYTHING.md
│   └── SYNTHETIC_DATA.md
│
├── experiments/
├── tests/
├── config.py
├── generate_synthetic_data.py
├── DATA_CONTRACT.md
└── README.md
```

---

## 5. Prerequisites

Install the following software before starting:

| Software | Purpose |
|---|---|
| Git | Repository cloning |
| Python | Backend and ML execution |
| Node.js and npm | Frontend execution |
| VS Code | Recommended development environment |
| BioGears | Physiological simulation pathway |

Use a Python version compatible with the dependencies in `backend/requirements.txt`.

---

## 6. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/DakshYadav1/ClinTwin-Sentinel.git
```

Navigate into the project:

```bash
cd ClinTwin-Sentinel
```

### Step 2: Set up the Python environment

Create a virtual environment:

**Windows PowerShell**

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install backend dependencies

From the project root:

```bash
pip install -r backend/requirements.txt
```

### Step 4: Install frontend dependencies

Open another terminal:

```bash
cd frontend
npm install
```

---

## 7. Running the Application

The backend and frontend must run simultaneously in separate terminals.

### Terminal 1: Start FastAPI Backend

From the project root, with the virtual environment activated:

```bash
uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2: Start React Frontend

From the frontend directory:

```bash
npm run dev
```

Vite normally starts the application at:

```text
http://localhost:5173
```

Open this address in your browser to access the ClinTwin Sentinel dashboard.

---

## 8. MIMIC-IV Setup

**Important:** MIMIC-IV is not distributed with this repository.

Users must independently obtain authorized access to MIMIC-IV and comply with its data-use requirements.

### Required local files

The MIMIC-IV workflow expects the processed dataset and trained checkpoints locally.

Expected paths include:

```text
data/
└── real/
    ├── raw/
    │   └── mimiciv-3.1/
    │
    └── processed/
        └── splits/
            └── mimic_test.csv

models/
└── mimic/
    ├── mimic_bilstm.pt
    └── mimic_transformer.pt
```

These files are intentionally excluded from GitHub.

### Reproducing the pipeline

The repository includes scripts under:

```text
scripts/mimic/
```

They cover:

1. ICU cohort construction.
2. Laboratory and vital-sign extraction and cleaning.
3. SOFA feature engineering.
4. Vasopressor feature engineering.
5. Deterioration label generation.
6. Patient-level dataset splitting.
7. Six-hour sequence construction.
8. Temporal model training.
9. Model evaluation.

Refer to:

- `docs/MIMIC_IV_SETUP.md`
- `DATA_CONTRACT.md`

The MIMIC dashboard requires the processed test dataset and both trained checkpoints to be generated locally before it can return predictions.

### MIMIC-IV API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/mimic/roster` | Retrieves retrospective ICU cohort |
| `GET /api/mimic/patients/{stay_id}` | Retrieves selected ICU stay and predictions |
| `GET /api/mimic/patients/{stay_id}/trajectory` | Retrieves prediction trajectory |

The MIMIC-IV pathway is retrospective and does not process live clinical data.

---

## 9. BioGears Setup

The BioGears pathway requires a compatible BioGears installation and the appropriate local simulation configuration.

The repository includes BioGears model artifacts:

```text
models/
├── biogears_lstm_model.pt
├── biogears_transformer_model.pt
└── biogears_scaler.joblib
```

Additional legacy model artifacts or configuration files may be required by the existing backend startup and inference workflow. These are not all distributed with the repository.

Follow the setup and architecture documentation:

- `docs/BIOGEARS_IMPLEMENTATION.md`
- `docs/BIOGEARS_INTEGRATION_BOUNDARY.md`
- `WHATIF_ARCHITECTURE.md`

---

## 10. Model Evaluation

The MIMIC-IV temporal models were evaluated on a held-out patient-level test set.

| Metric | BiLSTM | Transformer |
|---|---:|---:|
| AUROC | 0.6567 | 0.7186 |
| AUPRC | 0.2834 | 0.3224 |
| Precision | 0.2611 | 0.2495 |
| Recall | 0.4224 | 0.7519 |
| F1-score | 0.3227 | 0.3747 |

The Transformer achieved higher recall and AUROC in this evaluation, while both models demonstrated limitations, including false positives.

These are retrospective research results and do not establish clinical effectiveness.

---

## 11. Important Research Limitations

- MIMIC-IV is retrospective, not a live clinical data source.
- Hourly SOFA-related features are research proxies; conventional SOFA scoring generally uses the worst values over a 24-hour period.
- Missing physiological measurements and incomplete SOFA component coverage affect the available cohort.
- Prediction probabilities are not clinical diagnoses.
- Both models require further external validation and clinical evaluation.
- What-If interventions are simulated scenarios, not treatment recommendations.

**ClinTwin Sentinel is intended exclusively for research, education, and software prototyping.**

---

## 12. Troubleshooting

| Issue | Suggested action |
|---|---|
| Python package installation fails | Check Python compatibility and `backend/requirements.txt` |
| `uvicorn` not found | Activate the virtual environment and install backend requirements |
| Frontend dependencies missing | Run `npm install` inside `frontend/` |
| Frontend cannot reach backend | Verify that FastAPI is running on port 8000 |
| MIMIC roster fails to load | Verify the processed test dataset exists |
| MIMIC model loading fails | Verify both checkpoints exist under `models/mimic/` |
| BioGears simulation fails | Verify BioGears installation and scenario configuration |
| Model checkpoint missing | Generate or provide the required local model artifacts |

---

## 13. Documentation

Additional project documentation is available in the `docs/` directory.

- `DATA_CONTRACT.md` — Dataset and feature contract.
- `PRE_BIOGEARS_ARCHITECTURE.md` — Earlier system architecture.
- `WHATIF_ARCHITECTURE.md` — What-If simulation architecture.
- `docs/MIMIC_IV_SETUP.md` — MIMIC-IV setup.
- `docs/BIOGEARS_IMPLEMENTATION.md` — BioGears implementation.
- `docs/RUN_EVERYTHING.md` — Project execution guidance.

---

## 14. License and Data Usage

MIMIC-IV access is subject to the dataset's own access requirements and data-use agreement. Users must obtain authorization independently.

BioGears and other third-party components remain subject to their respective licenses and terms.

No raw MIMIC-IV patient data or MIMIC-trained model checkpoints are distributed in this repository.

---

## 15. Authors and Project

**Project:** ClinTwin Sentinel — ICU Digital Twin Research Prototype

**GitHub:** [DakshYadav1/ClinTwin-Sentinel](https://github.com/DakshYadav1/ClinTwin-Sentinel)

**Core technologies:** BioGears, MIMIC-IV, BiLSTM, Transformer, FastAPI, React, PyTorch.

