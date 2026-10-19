"""Download the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m src.download_data            # download if missing
    python -m src.download_data --force    # re-download

Primary source is the UCI archive file ``processed.cleveland.data``. If that
fails (network or server issue) we fall back to the ``ucimlrepo`` package,
which fetches the same dataset (UCI id 45).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
UCI_DATASET_ID = 45
EXPECTED_ROWS = 303
TIMEOUT_SECONDS = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_from_url(dest: Path) -> None:
    logger.info("Downloading %s", UCI_URL)
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "heart-mlops/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        dest.write_bytes(response.read())


def _download_with_ucimlrepo(dest: Path) -> None:
    logger.info("Falling back to ucimlrepo (id=%s)", UCI_DATASET_ID)
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional dependency

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    frame = frame[config.COLUMNS]  # enforce column order
    frame.to_csv(dest, header=False, index=False, na_rep="?")


def _verify(path: Path) -> None:
    frame = pd.read_csv(path, header=None, na_values=["?"])
    if frame.shape[1] != len(config.COLUMNS):
        raise ValueError(f"Expected {len(config.COLUMNS)} columns, found {frame.shape[1]}")
    if len(frame) != EXPECTED_ROWS:
        logger.warning("Expected %d rows but found %d", EXPECTED_ROWS, len(frame))


def download(force: bool = False, dest: Path | None = None) -> Path:
    """Download the raw dataset and return its path."""
    dest = Path(dest) if dest else config.RAW_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("Raw data already present at %s (use --force to refresh)", dest)
        return dest

    try:
        _download_from_url(dest)
        _verify(dest)
    except Exception as exc:  # noqa: BLE001 - we deliberately try the fallback
        logger.warning("Direct download failed (%s)", exc)
        try:
            _download_with_ucimlrepo(dest)
            _verify(dest)
        except Exception as fallback_exc:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the dataset from UCI. Download "
                f"'{UCI_URL}' manually and save it as '{dest}'."
            ) from fallback_exc

    logger.info("Saved %s (sha256=%s)", dest, sha256(dest))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        download(force=args.force)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
