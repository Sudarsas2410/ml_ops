"""Tests for the command-line entry points and the EDA script (all offline, in tmp dirs)."""

import json

import pandas as pd
import pytest

from src import config, data_processing, eda, predict, train
from tests.helpers import write_raw_file


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    """Point every project path at a temp directory so tests never touch real data."""
    raw = write_raw_file(tmp_path / "data" / "raw" / "processed.cleveland.data", n=303, seed=11)
    monkeypatch.setattr(config, "RAW_FILE", raw)
    monkeypatch.setattr(config, "CLEAN_FILE", tmp_path / "data" / "processed" / "heart_clean.csv")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "METRICS_FILE", tmp_path / "models" / "metrics.json")
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlartifacts").as_uri())
    return tmp_path


def test_prepare_dataset_builds_and_caches_clean_csv(isolated_paths):
    df = data_processing.prepare_dataset()
    assert config.CLEAN_FILE.exists()
    assert df.isna().sum().sum() == 0 and config.TARGET in df.columns
    cached = data_processing.prepare_dataset()  # second call reads the saved CSV
    pd.testing.assert_frame_equal(df, cached, check_dtype=False)


def test_data_processing_cli(isolated_paths, capsys):
    assert data_processing.main(["--force"]) == 0
    assert "Clean dataset: 303 rows x 14 columns" in capsys.readouterr().out


def test_eda_writes_all_figures_and_summary(isolated_paths):
    summary = eda.run_eda()
    figures = {p.name for p in config.FIGURES_DIR.glob("*.png")}
    assert {
        "00_missing_values.png",
        "01_class_balance.png",
        "02_numeric_histograms.png",
        "03_boxplots_by_target.png",
        "04_categorical_counts.png",
        "05_correlation_heatmap.png",
        "06_target_correlation.png",
    } <= figures
    saved = json.loads((config.REPORTS_DIR / "eda_summary.json").read_text())
    assert saved["rows_raw"] == 303 and saved["n_features"] == 13
    assert set(saved["class_balance"]) == {"No disease", "Disease"}
    assert summary["missing_raw"] == {"ca": 4, "thal": 2}


def test_train_cli_end_to_end(isolated_paths, capsys):
    clean_csv = data_processing.save_clean(data_processing.clean(data_processing.load_raw()))
    code = train.main(
        [
            "--data",
            str(clean_csv),
            "--models",
            "logistic_regression",
            "--cv-folds",
            "3",
            "--tracking-uri",
            f"sqlite:///{isolated_paths / 'mlflow.db'}",
            "--no-register",
        ]
    )
    assert code == 0
    assert config.MODEL_FILE.exists() and config.METRICS_FILE.exists()
    out = capsys.readouterr().out
    assert "logistic_regression" in out and "<- selected" in out
    assert json.loads(config.METRICS_FILE.read_text())["data_sha256"]


def test_predict_cli_json_file_and_csv(isolated_paths, model_file, capsys):
    assert predict.main(["--json", json.dumps(config.SAMPLE_PATIENT), "--model", str(model_file)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["prediction"] in (0, 1)

    sample = isolated_paths / "sample.json"
    sample.write_text(json.dumps([config.SAMPLE_PATIENT, config.SAMPLE_PATIENT]))
    assert predict.main(["--file", str(sample), "--model", str(model_file)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2

    clean_csv = data_processing.save_clean(data_processing.clean(data_processing.load_raw()))
    assert predict.main(["--csv", str(clean_csv), "--limit", "3", "--model", str(model_file)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 3
