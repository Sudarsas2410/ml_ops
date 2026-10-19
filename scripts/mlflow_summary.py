"""Summarise the latest MLflow comparison run as Markdown (for the report and CI).

Usage:
    python scripts/mlflow_summary.py                         # writes reports/experiment_summary.md
    python scripts/mlflow_summary.py --output "$GITHUB_STEP_SUMMARY" --append
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _f(value, digits: int = 3) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}f}"


def build_summary(tracking_uri: str, experiment: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=[experiment], output_format="pandas", order_by=["attributes.start_time DESC"]
    )
    if runs.empty:
        return f"# Experiment summary\n\nNo runs found in experiment `{experiment}`.\n"

    parents = runs[runs["tags.stage"] == "comparison"]
    parent = parents.iloc[0]
    children = runs[runs["tags.mlflow.parentRunId"] == parent["run_id"]].sort_values("params.model")

    cv_rows, test_rows = [], []
    for _, r in children.iterrows():
        name = r["params.model"]
        star = " ★" if name == parent["tags.best_model"] else ""
        cv_rows.append(
            [f"{name}{star}"]
            + [
                f"{_f(r[f'metrics.cv_{m}_mean'])} ± {_f(r[f'metrics.cv_{m}_std'])}"
                for m in ("accuracy", "precision", "recall", "roc_auc")
            ]
        )
        test_rows.append(
            [f"{name}{star}"]
            + [_f(r[f"metrics.test_{m}"]) for m in ("accuracy", "precision", "recall", "f1", "roc_auc")]
            + [_f(r["metrics.fit_seconds"], 1)]
        )

    params_md = []
    for _, r in children.iterrows():
        best = {
            c.removeprefix("params.best_"): r[c]
            for c in children.columns
            if c.startswith("params.best_") and r[c] is not None and r[c] == r[c]
        }
        params_md.append(f"- **{r['params.model']}**: " + ", ".join(f"`{k}={v}`" for k, v in best.items()))

    return "\n".join(
        [
            "# Experiment tracking summary",
            "",
            f"- Experiment: `{experiment}`  |  tracking URI: `{tracking_uri}`",
            f"- Runs logged (all sessions): **{len(runs)}**  |  latest comparison run: `{parent['run_id']}`",
            f"- Seed {parent.get('params.seed')}, {parent.get('params.cv_folds')}-fold stratified CV, "
            f"hold-out test size {parent.get('params.test_size')}",
            f"- Selected model (highest CV ROC-AUC): **{parent['tags.best_model']}**",
            "",
            "## Cross-validation metrics (mean ± std over folds, training split)",
            "",
            _table(["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"], cv_rows),
            "",
            "## Hold-out test metrics (never used for tuning or selection)",
            "",
            _table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Fit time (s)"], test_rows),
            "",
            "## Selected hyper-parameters (GridSearchCV, refit on ROC-AUC)",
            "",
            *params_md,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tracking-uri", default=config.MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=config.REPORTS_DIR / "experiment_summary.md")
    parser.add_argument("--append", action="store_true", help="append instead of overwrite")
    args = parser.parse_args()

    # The summary contains non-ASCII characters (star, plus-minus). Windows consoles and files
    # default to cp1252, so force UTF-8 for both stdout and the output file.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    text = build_summary(args.tracking_uri, args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.append else "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
