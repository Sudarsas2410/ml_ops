"""Inference helpers shared by the CLI and the API.

Usage:
    python -m src.predict --json '{"age": 63, "sex": 1, ...}'
    python -m src.predict --file scripts/sample_input.json
    python -m src.predict --csv data/processed/heart_clean.csv --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config

LABELS = {0: "no heart disease", 1: "heart disease"}


def load_model(path: Path | str | None = None):
    """Load the saved preprocessing + model pipeline."""
    path = Path(path) if path else config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train one with `python -m src.train`.")
    return joblib.load(path)


def to_frame(records: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Turn one or many patient dicts into a float64 frame in training column order."""
    if isinstance(records, dict):
        records = [records]
    frame = pd.DataFrame(records)
    missing = [c for c in config.FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")
    return frame[config.FEATURES].astype("float64")


def predict(model, records: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return prediction, label, confidence and disease probability per record.

    ``confidence`` is the model's probability for the *predicted* class;
    ``probability_disease`` is always P(heart disease).
    """
    frame = to_frame(records)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        label = int(p >= 0.5)
        results.append(
            {
                "prediction": label,
                "label": LABELS[label],
                "confidence": round(float(p if label else 1 - p), 4),
                "probability_disease": round(float(p), 4),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="a patient record as a JSON string")
    source.add_argument("--file", type=Path, help="JSON file with one record or a list")
    source.add_argument("--csv", type=Path, help="CSV file with the feature columns")
    parser.add_argument("--limit", type=int, default=10, help="max CSV rows to score")
    parser.add_argument("--model", type=Path, default=config.MODEL_FILE)
    args = parser.parse_args(argv)

    if args.json:
        records = json.loads(args.json)
    elif args.file:
        records = json.loads(args.file.read_text())
    else:
        records = pd.read_csv(args.csv).head(args.limit)[config.FEATURES].to_dict("records")

    print(json.dumps(predict(load_model(args.model), records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
