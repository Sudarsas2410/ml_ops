import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_processing import (
    build_preprocessor,
    clean,
    get_feature_names,
    load_raw,
    missing_value_report,
    split_xy,
    validate_schema,
)
from tests.helpers import make_raw_frame, write_raw_file


# ------------------------------- loading -------------------------------- #
def test_load_raw_parses_question_marks_as_missing(tmp_path):
    path = write_raw_file(tmp_path / "raw.data", n=120, seed=3)
    df = load_raw(path)
    assert list(df.columns) == config.COLUMNS
    assert len(df) == 120
    assert df["ca"].isna().sum() == 4 and df["thal"].isna().sum() == 2
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in config.COLUMNS)


def test_load_raw_missing_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nope.data")


def test_validate_schema():
    df = make_raw_frame(20)
    validate_schema(df.drop(columns="num"))
    with pytest.raises(ValueError, match="age"):
        validate_schema(df.drop(columns="age"))
    with pytest.raises(ValueError, match="target"):
        validate_schema(df.drop(columns="num"), require_target=True)


# ------------------------------- cleaning ------------------------------- #
def test_clean_binarises_target():
    raw = make_raw_frame(50, seed=1)
    raw["num"] = [0, 1, 2, 3, 4] * 10
    out = clean(raw.drop_duplicates(subset=config.FEATURES))
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert config.RAW_TARGET not in out.columns


def test_clean_target_matches_rule(raw_df, clean_df):
    assert clean_df[config.TARGET].sum() == int((raw_df["num"] > 0).sum())


def test_clean_imputes_all_missing(raw_df, clean_df):
    assert raw_df.isna().sum().sum() > 0
    assert clean_df.isna().sum().sum() == 0


def test_clean_uses_mode_for_categorical_and_median_for_numeric():
    raw = make_raw_frame(40, seed=2, missing=False)
    raw["ca"] = 0.0
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "chol"] = np.nan
    expected_median = raw["chol"].median()
    out = clean(raw)
    assert out.loc[0, "ca"] == 0
    assert out.loc[1, "chol"] == pytest.approx(expected_median)


def test_clean_drops_duplicates_and_unlabelled_rows():
    raw = make_raw_frame(30, seed=4, missing=False)
    noisy = pd.concat([raw, raw.iloc[:5]], ignore_index=True)
    noisy.loc[10, "num"] = np.nan  # not one of the duplicated rows
    out = clean(noisy)
    assert len(out) == 30 - 1  # 5 duplicates removed, 1 row without label removed


def test_clean_masks_implausible_values_then_imputes():
    raw = make_raw_frame(40, seed=5, missing=False)
    raw.loc[3, "chol"] = 9999
    raw.loc[4, "cp"] = 9
    out = clean(raw)
    assert out["chol"].max() <= config.VALID_RANGES["chol"][1]
    assert set(out["cp"]) <= config.VALID_CATEGORIES["cp"]


def test_clean_without_imputation_keeps_nans():
    out = clean(make_raw_frame(60, seed=6), impute=False)
    assert out.isna().sum().sum() > 0


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing"):
        clean(make_raw_frame(10).drop(columns=["chol"]))


def test_clean_does_not_mutate_input():
    raw = make_raw_frame(30)
    snapshot = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_missing_value_report(raw_df):
    report = missing_value_report(raw_df)
    assert set(report.index) == {"ca", "thal"}
    assert report.loc["ca", "missing"] == 4


# ------------------------------- features ------------------------------- #
def test_split_xy_shapes_and_dtypes(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.FEATURES
    assert X.shape == (len(clean_df), 13)
    assert (X.dtypes == "float64").all()
    assert set(y.unique()) <= {0, 1}


def test_preprocessor_output_is_finite_and_scaled(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor()
    Z = pre.fit_transform(X)
    assert np.isfinite(Z).all()
    n_onehot = sum(X[c].nunique() for c in config.CATEGORICAL_FEATURES)
    assert Z.shape == (len(X), len(config.NUMERIC_FEATURES) + n_onehot)
    numeric_block = Z[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_preprocessor_handles_missing_and_unseen_values(clean_df):
    X, _ = split_xy(clean_df)
    pre = build_preprocessor().fit(X)
    odd = X.head(3).copy()
    odd.loc[odd.index[0], "chol"] = np.nan  # missing numeric
    odd.loc[odd.index[1], "thal"] = 5.0  # category never seen in training
    odd.loc[odd.index[2], "ca"] = np.nan  # missing categorical
    assert np.isfinite(pre.transform(odd)).all()


def test_feature_names_match_matrix_width(trained_pipeline, clean_df):
    X, _ = split_xy(clean_df)
    names = get_feature_names(trained_pipeline)
    assert len(names) == trained_pipeline.named_steps["preprocess"].transform(X).shape[1]
    assert "age" in names and any(n.startswith("cp_") for n in names)
