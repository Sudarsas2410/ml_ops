"""Data loading, cleaning and the sklearn preprocessing pipeline.

Cleaning (``clean``) produces the processed CSV: types coerced, duplicates
dropped, implausible values masked, the 0-4 ``num`` label turned into a binary
``target`` and the handful of missing values imputed.

The ``build_preprocessor`` transformer repeats imputation, scaling and one-hot
encoding *inside* the model pipeline. That keeps training and inference
identical and means missing values are handled safely at serving time too.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the headerless UCI file; '?' marks missing values."""
    path = Path(path) if path else config.RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run `python -m src.download_data` first.")
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values=["?"], skipinitialspace=True)


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """Raise ``ValueError`` if expected columns are missing."""
    required = list(config.FEATURES)
    if require_target:
        required.append(config.TARGET)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and percentage of missing values (only columns with any)."""
    counts = df.isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": (counts / len(df) * 100).round(2)})
    return report[report["missing"] > 0]


def _mask_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Set out-of-range values to NaN so they are imputed rather than trusted."""
    for col, (low, high) in config.VALID_RANGES.items():
        bad = ~df[col].between(low, high) & df[col].notna()
        if bad.any():
            logger.warning("Masking %d implausible value(s) in '%s'", int(bad.sum()), col)
            df.loc[bad, col] = float("nan")
    for col, allowed in config.VALID_CATEGORIES.items():
        bad = ~df[col].isin(allowed) & df[col].notna()
        if bad.any():
            logger.warning("Masking %d invalid category value(s) in '%s'", int(bad.sum()), col)
            df.loc[bad, col] = float("nan")
    return df


def clean(df: pd.DataFrame, impute: bool = True) -> pd.DataFrame:
    """Return a cleaned copy with a binary ``target`` column.

    Steps: coerce to numeric, drop duplicates, drop rows without a label,
    binarise label (num > 0), mask implausible values, impute (median for
    continuous, mode for categorical).
    """
    missing_cols = [c for c in config.COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out = df[config.COLUMNS].copy()
    for col in config.COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.drop_duplicates().dropna(subset=[config.RAW_TARGET]).reset_index(drop=True)
    if len(out) < before:
        logger.info("Dropped %d duplicate/unlabelled row(s)", before - len(out))

    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    out = out.drop(columns=[config.RAW_TARGET])
    out = _mask_implausible(out)

    if impute:
        for col in config.NUMERIC_FEATURES:
            out[col] = out[col].fillna(out[col].median())
        for col in config.CATEGORICAL_FEATURES:
            if out[col].isna().any():
                out[col] = out[col].fillna(out[col].mode().iloc[0])
            out[col] = out[col].astype(int)
    return out


def save_clean(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    path = Path(path) if path else config.CLEAN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved cleaned data (%d rows) to %s", len(df), path)
    return path


def prepare_dataset(force: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset, building it from the raw file if necessary."""
    if config.CLEAN_FILE.exists() and not force:
        return pd.read_csv(config.CLEAN_FILE)
    if not config.RAW_FILE.exists():
        from src.download_data import download

        download()
    cleaned = clean(load_raw())
    save_clean(cleaned)
    return cleaned


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into float64 feature matrix (stable schema for MLflow) and target."""
    validate_schema(df, require_target=True)
    X = df[config.FEATURES].astype("float64")
    y = df[config.TARGET].astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after preprocessing from a fitted ``preprocess``+``model`` pipeline."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cleaned dataset CSV")
    parser.add_argument("--force", action="store_true", help="rebuild even if it exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = prepare_dataset(force=args.force)
    print(f"Clean dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df[config.TARGET].value_counts().rename({0: "no disease", 1: "disease"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
