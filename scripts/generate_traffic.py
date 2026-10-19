"""Send realistic traffic to the API so the Grafana dashboard has something to show.

Usage:
    python scripts/generate_traffic.py --url http://localhost:8000 --requests 300 --delay 0.05

Uses real rows from data/processed/heart_clean.csv when available (otherwise random
valid records) and mixes in a few invalid payloads so error panels light up too.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_records() -> list[dict]:
    path = ROOT / "data" / "processed" / "heart_clean.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return [{k: float(row[k]) for k in FEATURES} for row in csv.DictReader(handle)]


def random_record(rng: random.Random) -> dict:
    return {
        "age": rng.randint(30, 77),
        "sex": rng.randint(0, 1),
        "cp": rng.randint(1, 4),
        "trestbps": rng.randint(94, 200),
        "chol": rng.randint(126, 564),
        "fbs": rng.randint(0, 1),
        "restecg": rng.choice([0, 1, 2]),
        "thalach": rng.randint(71, 202),
        "exang": rng.randint(0, 1),
        "oldpeak": round(rng.uniform(0, 6), 1),
        "slope": rng.randint(1, 3),
        "ca": rng.randint(0, 3),
        "thal": rng.choice([3, 6, 7]),
    }


def post(url: str, payload: dict) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--invalid-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_records()
    print(
        f"Sending {args.requests} requests to {args.url}/predict "
        f"({'real rows' if records else 'random records'})"
    )
    statuses: Counter = Counter()
    for i in range(args.requests):
        payload = rng.choice(records) if records else random_record(rng)
        if rng.random() < args.invalid_rate:
            payload = {**payload, "age": 999}  # triggers a 422
        statuses[post(f"{args.url}/predict", payload)] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} sent")
        time.sleep(args.delay)
    print("Done. Status codes:", dict(statuses))


if __name__ == "__main__":
    main()
