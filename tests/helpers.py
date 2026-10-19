"""Synthetic stand-in for the UCI file, used ONLY by the unit tests.

It mimics the schema of ``processed.cleveland.data`` (14 columns, '?' for
missing values, ``num`` in 0-4) so tests run offline and deterministically.
It is not real patient data and must never be used for reported results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def make_raw_frame(n: int = 300, seed: int = 0, missing: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.choice([1, 2, 3, 4], n, p=[0.1, 0.17, 0.28, 0.45])
    trestbps = rng.normal(131, 17, n).clip(94, 200).round()
    chol = rng.normal(246, 51, n).clip(126, 564).round()
    fbs = (rng.random(n) < 0.15).astype(int)
    restecg = rng.choice([0, 1, 2], n, p=[0.5, 0.01, 0.49])
    thalach = (205 - 0.9 * age + rng.normal(0, 18, n)).clip(71, 202).round()
    exang = (rng.random(n) < 0.33).astype(int)
    oldpeak = np.abs(rng.gamma(1.0, 1.0, n)).clip(0, 6.2).round(1)
    slope = rng.choice([1, 2, 3], n, p=[0.47, 0.46, 0.07])
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = rng.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39])

    latent = (
        0.04 * (age - 54)
        + 0.9 * sex
        + 1.1 * (cp == 4)
        - 0.03 * (thalach - 150)
        + 0.6 * oldpeak
        + 0.8 * ca
        + 1.0 * (thal == 7)
        + 0.7 * exang
        + rng.normal(0, 0.8, n)
    )
    num = np.digitize(latent, [1.9, 3.0, 3.8, 4.5])  # 0-4

    frame = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca.astype(float),
            "thal": thal.astype(float),
            "num": num,
        }
    )[config.COLUMNS]
    if missing:
        frame.loc[rng.choice(n, 4, replace=False), "ca"] = np.nan
        frame.loc[rng.choice(n, 2, replace=False), "thal"] = np.nan
    return frame


def write_raw_file(path: Path, n: int = 300, seed: int = 0) -> Path:
    """Write a headerless CSV that uses '?' for missing values, like the UCI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    make_raw_frame(n, seed).to_csv(path, header=False, index=False, na_rep="?")
    return path
