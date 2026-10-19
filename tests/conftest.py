"""Shared fixtures. All data is synthetic (see tests/helpers.py): tests run offline."""

from __future__ import annotations

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import joblib  # noqa: E402
import pytest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data_processing import build_preprocessor, clean, split_xy  # noqa: E402
from tests.helpers import make_raw_frame  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_raw_frame(300, seed=0)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_pipeline(clean_df):
    X, y = split_xy(clean_df)
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000, random_state=0))]
    )
    return pipeline.fit(X, y)


@pytest.fixture()
def model_file(tmp_path, trained_pipeline):
    """A saved pipeline plus metrics.json, laid out like the real models/ directory."""
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "best_model": "logistic_regression",
                "version": "test",
                "cv": {"roc_auc": 0.9},
                "test": {"roc_auc": 0.9},
                "trained_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    return path
