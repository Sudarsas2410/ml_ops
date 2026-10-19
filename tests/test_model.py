import json

import joblib
import mlflow
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src import config
from src.data_processing import build_preprocessor, split_xy
from src.predict import load_model, predict, to_frame
from src.train import evaluate, model_specs, run_training


def test_predict_proba_is_valid_probability(trained_pipeline, clean_df):
    X, _ = split_xy(clean_df)
    proba = trained_pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_beats_chance_on_training_signal(trained_pipeline, clean_df):
    X, y = split_xy(clean_df)
    assert evaluate(y, trained_pipeline.predict(X), trained_pipeline.predict_proba(X)[:, 1])["roc_auc"] > 0.8


def test_predict_returns_documented_schema(trained_pipeline):
    result = predict(trained_pipeline, config.SAMPLE_PATIENT)
    assert len(result) == 1
    item = result[0]
    assert set(item) == {"prediction", "label", "confidence", "probability_disease"}
    assert item["prediction"] in (0, 1)
    assert 0.5 <= item["confidence"] <= 1.0
    assert 0.0 <= item["probability_disease"] <= 1.0


def test_confidence_is_probability_of_predicted_class(trained_pipeline):
    item = predict(trained_pipeline, config.SAMPLE_PATIENT)[0]
    p = item["probability_disease"]
    assert item["confidence"] == pytest.approx(p if item["prediction"] == 1 else 1 - p, abs=1e-4)


def test_batch_prediction(trained_pipeline, clean_df):
    rows = clean_df[config.FEATURES].head(5).to_dict("records")
    assert len(predict(trained_pipeline, rows)) == 5


def test_to_frame_orders_columns_and_validates():
    shuffled = dict(reversed(list(config.SAMPLE_PATIENT.items())))
    assert list(to_frame(shuffled).columns) == config.FEATURES
    with pytest.raises(ValueError, match="Missing feature"):
        to_frame({"age": 50})


def test_saved_model_round_trip_gives_identical_predictions(tmp_path, trained_pipeline, clean_df):
    X, _ = split_xy(clean_df)
    path = tmp_path / "m.joblib"
    joblib.dump(trained_pipeline, path)
    reloaded = load_model(path)
    assert np.array_equal(trained_pipeline.predict_proba(X), reloaded.predict_proba(X))


def test_load_model_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model(tmp_path / "missing.joblib")


def test_training_is_reproducible_with_fixed_seed(clean_df):
    X, y = split_xy(clean_df)

    def fit():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model_specs(7)["random_forest"][0])])
        return pipe.fit(X, y).predict_proba(X)

    assert np.array_equal(fit(), fit())


def test_evaluate_perfect_and_worst_cases():
    y = np.array([0, 0, 1, 1])
    perfect = evaluate(y, y, y.astype(float))
    assert all(v == 1.0 for v in perfect.values())
    worst = evaluate(y, 1 - y, 1 - y.astype(float))
    assert worst["accuracy"] == 0.0 and worst["roc_auc"] == 0.0


def test_model_specs_cover_required_models():
    specs = model_specs(0)
    assert {"logistic_regression", "random_forest"} <= set(specs)
    for estimator, grid in specs.values():
        assert grid and all(k.startswith("model__") for k in grid)


def test_run_training_unknown_model_rejected(clean_df, tmp_path):
    with pytest.raises(ValueError, match="Unknown model"):
        run_training(clean_df, models=["svm"], tracking_uri=f"sqlite:///{tmp_path / 'm.db'}")


def test_run_training_end_to_end_logs_to_mlflow(clean_df, tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    summary = run_training(
        clean_df,
        models=["logistic_regression", "random_forest"],
        cv_folds=3,
        model_path=tmp_path / "out" / "model.joblib",
        metrics_path=tmp_path / "out" / "metrics.json",
        figures_dir=tmp_path / "figs",
        tracking_uri=uri,
        experiment="pytest-experiment",
        artifact_root=(tmp_path / "artifacts").as_uri(),
        register=False,
    )

    # Outputs on disk
    assert (tmp_path / "out" / "model.joblib").exists()
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert saved["best_model"] in {"logistic_regression", "random_forest"}
    assert saved["test"]["roc_auc"] > 0.7
    assert set(saved["features"]) == set(config.FEATURES)
    assert (tmp_path / "figs" / "roc_overlay.png").exists()
    assert summary["best_model"] == saved["best_model"]

    # The saved artifact must serve predictions
    assert (
        0
        <= predict(load_model(tmp_path / "out" / "model.joblib"), config.SAMPLE_PATIENT)[0]["confidence"]
        <= 1
    )

    # MLflow: parent run + one child per model, each with params, metrics and plot artifacts
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["pytest-experiment"], output_format="pandas")
    children = runs[runs["tags.stage"] == "experiment"]
    assert len(children) == 2 and len(runs) == 3
    for col in ("metrics.cv_roc_auc_mean", "metrics.test_roc_auc", "metrics.test_recall", "params.model"):
        assert children[col].notna().all()
    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(children.iloc[0]["run_id"])}
    assert {"plots", "tuning", "reports"} <= artifacts
