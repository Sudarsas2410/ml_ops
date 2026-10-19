"""Exploratory data analysis: figures and a JSON summary.

Usage:
    python -m src.eda

Figures are written to ``reports/figures`` (override with HEART_REPORTS_DIR).
Each function returns the saved path so notebooks can display them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data_processing import load_raw, missing_value_report, prepare_dataset  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # silence category-axis chatter

PALETTE = {0: "#4C78A8", 1: "#E4572E"}  # colour-blind friendly pair
LABELS = {0: "No disease", 1: "Disease"}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def _save(fig: plt.Figure, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_class_balance(df: pd.DataFrame) -> Path:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 1, f"{value} ({value / counts.sum():.0%})", ha="center"
        )
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "01_class_balance.png")


def plot_numeric_histograms(df: pd.DataFrame) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(
            data=df,
            x=col,
            hue=config.TARGET,
            palette=PALETTE,
            bins=20,
            kde=True,
            element="step",
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{col}")
    for ax in axes.flat[len(cols) :]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[i]) for i in (0, 1)]
    axes.flat[-1].legend(handles, [LABELS[0], LABELS[1]], loc="center", frameon=False)
    fig.suptitle("Distribution of continuous features by class", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_numeric_histograms.png")


def plot_boxplots(df: pd.DataFrame) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 4))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET, palette=PALETTE, legend=False, ax=ax)
        ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
        ax.set_xlabel("")
    fig.suptitle("Continuous features by class (outlier check)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_boxplots_by_target.png")


def plot_categorical_counts(df: pd.DataFrame) -> Path:
    cols = config.CATEGORICAL_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        sns.countplot(data=df, x=col, hue=config.TARGET, palette=PALETTE, ax=ax, legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Categorical features by class", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_categorical_counts.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df.corr(numeric_only=True)
    mask = None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation heatmap (Pearson)")
    return _save(fig, "05_correlation_heatmap.png")


def plot_target_correlation(df: pd.DataFrame) -> Path:
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of each feature with disease")
    ax.set_xlabel("Pearson r")
    return _save(fig, "06_target_correlation.png")


def plot_missing_values(raw: pd.DataFrame) -> Path | None:
    report = missing_value_report(raw)
    if report.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(report.index, report["missing"], color="#8C6BB1")
    ax.set_title("Missing values in raw data")
    ax.set_ylabel("Count")
    return _save(fig, "00_missing_values.png")


def summarise(raw: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Machine-readable summary used in the report."""
    balance = clean_df[config.TARGET].value_counts().to_dict()
    corr = clean_df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    return {
        "rows_raw": int(len(raw)),
        "rows_clean": int(len(clean_df)),
        "n_features": len(config.FEATURES),
        "missing_raw": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "class_balance": {LABELS[int(k)]: int(v) for k, v in balance.items()},
        "positive_rate": round(float(clean_df[config.TARGET].mean()), 4),
        "top_correlations_with_target": corr.reindex(corr.abs().sort_values(ascending=False).index)
        .head(5)
        .round(3)
        .to_dict(),
        "numeric_describe": clean_df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
    }


def run_eda() -> dict:
    _style()
    raw = load_raw()
    clean_df = prepare_dataset()
    plot_missing_values(raw)
    plot_class_balance(clean_df)
    plot_numeric_histograms(clean_df)
    plot_boxplots(clean_df)
    plot_categorical_counts(clean_df)
    plot_correlation_heatmap(clean_df)
    plot_target_correlation(clean_df)
    summary = summarise(raw, clean_df)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run_eda(), indent=2))
