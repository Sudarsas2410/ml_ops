"""Central configuration: paths, schema and modelling constants.

Every path can be overridden with an environment variable so the same code
runs unchanged on a laptop, in GitHub Actions and inside a container.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("HEART_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.getenv("HEART_MODELS_DIR", ROOT_DIR / "models"))
REPORTS_DIR = Path(os.getenv("HEART_REPORTS_DIR", ROOT_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

# Experiment tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", (ROOT_DIR / "mlartifacts").as_uri())
MLFLOW_EXPERIMENT = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# Modelling
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Schema (processed.cleveland.data has no header row)
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
RAW_TARGET = "num"  # 0 = no disease, 1-4 = disease severity
TARGET = "target"  # binary: 0 = absence, 1 = presence
FEATURES = [c for c in COLUMNS if c != RAW_TARGET]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure in mm Hg on admission",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting ECG result (0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}

# Physiologically plausible ranges (min, max); also used by the API validators.
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 260),
    "chol": (80, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
}
VALID_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

# Example record used in docs, API schema and smoke tests
SAMPLE_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}
