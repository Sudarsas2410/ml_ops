"""Train, tune, evaluate and track heart-disease classifiers with MLflow.

Usage:
    python -m src.train                      # all three models
    python -m src.train --models random_forest
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Method
------
* Stratified 80/20 hold-out split. The test set is never used for tuning or
  for choosing the winner.
* Every model is a single sklearn ``Pipeline`` (preprocess -> estimator) tuned
  with ``GridSearchCV`` over 5-fold stratified CV, refit on ROC-AUC.
* CV metrics (mean +/- std) come from the folds of the best grid point; test
  metrics come from the untouched hold-out set.
* The winner is chosen by CV ROC-AUC and saved as ``models/model.joblib``.
* One parent MLflow run holds the comparison, with one nested child per model
  logging parameters, metrics, plots and the fitted pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # no outbound telemetry in CI/offline

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.data_processing import (
    build_preprocessor,
    get_feature_names,
    prepare_dataset,
    split_xy,
)  # noqa: E402,E501

logger = logging.getLogger("train")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRIC_NAMES = list(SCORING)
CLASS_NAMES = ["No disease", "Disease"]


def model_specs(seed: int) -> dict[str, tuple[object, dict]]:
    """Estimator and hyper-parameter grid for each candidate model."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


# --------------------------------------------------------------------------- #
# Metrics and plots
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_roc(name: str, y_true, y_proba, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_title(f"ROC curve (hold-out test set): {name}")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion(name: str, y_true, y_pred, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix: {name}")
    ax.grid(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_importance(name: str, pipeline: Pipeline, path: Path, top: int = 15) -> Path | None:
    model = pipeline.named_steps["model"]
    names = np.array(get_feature_names(pipeline))
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Feature importance"
    elif hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "Coefficient (standardised features)"
    else:
        return None
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    colors = ["#E4572E" if values[i] > 0 else "#4C78A8" for i in order]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(names[order], values[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top features: {name}")
    ax.set_xlabel(label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(results: list[dict], path: Path) -> Path:
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cv = [r["cv"]["roc_auc"] for r in results]
    cv_std = [r["cv_std"]["roc_auc"] for r in results]
    test = [r["test"]["roc_auc"] for r in results]
    ax.bar(x - width / 2, cv, width, yerr=cv_std, capsize=4, label="CV ROC-AUC (mean ± std)", color="#4C78A8")
    ax.bar(x + width / 2, test, width, label="Hold-out ROC-AUC", color="#E4572E")
    ax.set_xticks(x, names)
    ax.set_ylim(max(0.5, min(cv + test) - 0.1), 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_overlay(results: list[dict], y_test, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for r in results:
        RocCurveDisplay.from_predictions(
            y_test, r["test_proba"], name=f"{r['name']} (AUC {r['test']['roc_auc']:.3f})", ax=ax
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_title("ROC curves on hold-out test set")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# MLflow helpers
# --------------------------------------------------------------------------- #
def setup_mlflow(tracking_uri: str, experiment: str, artifact_root: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)


def log_sklearn_model(pipeline: Pipeline, X_train: pd.DataFrame, registered_name: str | None = None):
    """Log the fitted pipeline with signature and input example (MLflow 2 and 3 compatible)."""
    kwargs = {
        "sk_model": pipeline,
        "signature": infer_signature(X_train, pipeline.predict_proba(X_train)),
        "input_example": X_train.head(3),
        # MLflow 3 defaults to skops, which rejects some numpy types inside sklearn pipelines.
        "serialization_format": "cloudpickle",
    }
    if registered_name:
        kwargs["registered_model_name"] = registered_name
    try:
        return mlflow.sklearn.log_model(name="model", **kwargs)  # MLflow >= 3
    except TypeError:
        return mlflow.sklearn.log_model(artifact_path="model", **kwargs)  # MLflow 2.x


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one(name, estimator, grid, splits, cv, fig_dir: Path) -> dict:
    X_train, X_test, y_train, y_test = splits
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])

    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tags({"model_type": name, "stage": "experiment"})
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=SCORING,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        best: Pipeline = search.best_estimator_
        idx = search.best_index_
        cv_mean = {m: float(search.cv_results_[f"mean_test_{m}"][idx]) for m in METRIC_NAMES}
        cv_std = {m: float(search.cv_results_[f"std_test_{m}"][idx]) for m in METRIC_NAMES}

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test = evaluate(y_test, y_pred, y_proba)

        best_params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        mlflow.log_params(
            {
                "model": name,
                "cv_folds": cv.get_n_splits(),
                "grid_size": len(search.cv_results_["params"]),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features_in": X_train.shape[1],
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )
        mlflow.log_metrics({f"cv_{m}_mean": v for m, v in cv_mean.items()})
        mlflow.log_metrics({f"cv_{m}_std": v for m, v in cv_std.items()})
        mlflow.log_metrics({f"test_{m}": v for m, v in test.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)

        fig_dir.mkdir(parents=True, exist_ok=True)
        for plot in (
            plot_roc(name, y_test, y_proba, fig_dir / f"roc_{name}.png"),
            plot_confusion(name, y_test, y_pred, fig_dir / f"confusion_{name}.png"),
            plot_importance(name, best, fig_dir / f"importance_{name}.png"),
        ):
            if plot is not None:
                mlflow.log_artifact(str(plot), artifact_path="plots")

        cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc")
        cv_path = fig_dir / f"cv_results_{name}.csv"
        cv_table.to_csv(cv_path, index=False)
        mlflow.log_artifact(str(cv_path), artifact_path="tuning")
        report_path = fig_dir / f"classification_report_{name}.txt"
        report_path.write_text(
            classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        log_sklearn_model(best, X_train)

        logger.info(
            "%-20s CV ROC-AUC %.3f ± %.3f | test ROC-AUC %.3f | acc %.3f | recall %.3f",
            name,
            cv_mean["roc_auc"],
            cv_std["roc_auc"],
            test["roc_auc"],
            test["accuracy"],
            test["recall"],
        )
        return {
            "name": name,
            "run_id": run.info.run_id,
            "pipeline": best,
            "params": best_params,
            "cv": cv_mean,
            "cv_std": cv_std,
            "test": test,
            "test_proba": y_proba,
            "fit_seconds": fit_seconds,
        }


def run_training(
    data: pd.DataFrame,
    *,
    models: list[str] | None = None,
    seed: int = config.RANDOM_STATE,
    test_size: float = config.TEST_SIZE,
    cv_folds: int = config.CV_FOLDS,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    figures_dir: Path | None = None,
    tracking_uri: str | None = None,
    experiment: str = config.MLFLOW_EXPERIMENT,
    artifact_root: str | None = None,
    register: bool = True,
    data_sha256: str | None = None,
) -> dict:
    """Run the full experiment and return a summary dict (also saved as metrics.json)."""
    model_path = Path(model_path or config.MODEL_FILE)
    metrics_path = Path(metrics_path or config.METRICS_FILE)
    figures_dir = Path(figures_dir or config.FIGURES_DIR / "training")

    specs = model_specs(seed)
    selected = models or list(specs)
    unknown = set(selected) - set(specs)
    if unknown:
        raise ValueError(f"Unknown model(s) {sorted(unknown)}; choose from {sorted(specs)}")

    X, y = split_xy(data)
    splits = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_test, y_train, y_test = splits
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    setup_mlflow(
        tracking_uri or config.MLFLOW_TRACKING_URI,
        experiment,
        artifact_root or config.MLFLOW_ARTIFACT_ROOT,
    )

    with mlflow.start_run(run_name="model-comparison") as parent:
        mlflow.set_tags({"stage": "comparison", "dataset": "UCI Heart Disease (Cleveland)"})
        mlflow.log_params(
            {
                "seed": seed,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "models": ",".join(selected),
                "n_rows": len(data),
                "positive_rate": round(float(y.mean()), 4),
                "sklearn_version": sklearn.__version__,
                "python_version": platform.python_version(),
            }
        )
        if data_sha256:
            mlflow.set_tag("data_sha256", data_sha256)

        results = [train_one(name, *specs[name], splits, cv, figures_dir) for name in selected]
        best = max(results, key=lambda r: r["cv"]["roc_auc"])

        summary_plots = [
            plot_comparison(results, figures_dir / "model_comparison.png"),
            plot_roc_overlay(results, y_test, figures_dir / "roc_overlay.png"),
        ]
        for plot in summary_plots:
            mlflow.log_artifact(str(plot), artifact_path="plots")

        mlflow.set_tags({"best_model": best["name"], "best_child_run_id": best["run_id"]})
        mlflow.log_metrics({f"best_test_{m}": v for m, v in best["test"].items()})
        mlflow.log_metrics({f"best_cv_{m}_mean": v for m, v in best["cv"].items()})

        if register:
            try:
                log_sklearn_model(best["pipeline"], X_train, registered_name=config.REGISTERED_MODEL_NAME)
            except Exception as exc:  # noqa: BLE001 - registry is a bonus, never fail training
                logger.warning("Model registry step skipped: %s", exc)

        summary = {
            "best_model": best["name"],
            "selection_metric": "cv_roc_auc",
            "best_params": {
                k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                for k, v in best["params"].items()
            },
            "cv": {k: round(v, 4) for k, v in best["cv"].items()},
            "cv_std": {k: round(v, 4) for k, v in best["cv_std"].items()},
            "test": {k: round(v, 4) for k, v in best["test"].items()},
            "all_models": {
                r["name"]: {
                    "cv_roc_auc": round(r["cv"]["roc_auc"], 4),
                    "test_roc_auc": round(r["test"]["roc_auc"], 4),
                    "test_accuracy": round(r["test"]["accuracy"], 4),
                    "test_recall": round(r["test"]["recall"], 4),
                }
                for r in results
            },
            "features": list(config.FEATURES),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "seed": seed,
            "mlflow_parent_run_id": parent.info.run_id,
            "mlflow_best_run_id": best["run_id"],
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_sha256": data_sha256,
            "version": "1.0.0",
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best["pipeline"], model_path)
        metrics_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_dict(summary, "summary.json")
        mlflow.log_artifact(str(metrics_path))

    logger.info("Best model: %s -> %s", best["name"], model_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", help="subset of models to train")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--data", type=Path, help="cleaned CSV (default: data/processed/heart_clean.csv)")
    parser.add_argument("--tracking-uri", default=config.MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    parser.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("alembic", "matplotlib", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)  # pickle-format caution banner

    if args.data:
        data, data_path = pd.read_csv(args.data), args.data
    else:
        data, data_path = prepare_dataset(), config.CLEAN_FILE
    from src.download_data import sha256

    summary = run_training(
        data,
        models=args.models,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
        register=not args.no_register,
        data_sha256=sha256(Path(data_path)),
    )

    print("\n=== Results (hold-out test set) ===")
    for name, m in summary["all_models"].items():
        flag = "  <- selected" if name == summary["best_model"] else ""
        print(
            f"{name:<22} CV AUC {m['cv_roc_auc']:.3f} | test AUC {m['test_roc_auc']:.3f} "
            f"| acc {m['test_accuracy']:.3f} | recall {m['test_recall']:.3f}{flag}"
        )
    print(f"\nModel saved to {config.MODEL_FILE}")
    print(f"View runs: mlflow ui --backend-store-uri {args.tracking_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
