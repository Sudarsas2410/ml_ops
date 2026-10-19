"""Draw the system architecture diagram used in the report.

Usage:  python scripts/make_architecture_diagram.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

COLORS = {"data": "#DCE9F7", "ml": "#E3F1DF", "ci": "#FFF0D6", "serve": "#F3E3F5", "obs": "#FBE0DC"}
EDGES = {"data": "#4C78A8", "ml": "#5BA05B", "ci": "#D9962B", "serve": "#9A5BA5", "obs": "#D9614F"}


def box(ax, x, y, w, h, title, sub="", kind="ml"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=COLORS[kind],
            ec=EDGES[kind],
            lw=1.8,
        )
    )
    ax.text(
        x + w / 2,
        y + h * (0.62 if sub else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=8, color="#333")


def arrow(ax, p1, p2, label="", rad=0.0, dy=0.14):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color="#444",
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (p1[0] + p2[0]) / 2,
            (p1[1] + p2[1]) / 2 + dy,
            label,
            ha="center",
            fontsize=7.5,
            color="#222",
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9),
        )


def lane(ax, y, h, text):
    ax.add_patch(
        FancyBboxPatch(
            (0.15, y),
            15.7,
            h,
            boxstyle="round,pad=0.0,rounding_size=0.1",
            fc="#FAFAFA",
            ec="#BBB",
            lw=1,
            ls="--",
        )
    )
    ax.text(0.3, y + h - 0.2, text, fontsize=9, color="#555", fontweight="bold", va="top")


def elbow(ax, points, label="", label_at=0):
    """Polyline with an arrowhead on the last segment (keeps long links out of other boxes)."""
    xs, ys = zip(*points[:-1])
    ax.plot(
        xs + (points[-1][0],), ys + (points[-1][1],), color="#444", lw=1.4, zorder=1, solid_capstyle="round"
    )
    ax.add_patch(
        FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#444")
    )
    if label:
        a, b = points[label_at], points[label_at + 1]
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.13,
            label,
            ha="center",
            fontsize=7.5,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9),
        )


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    ax.text(
        8, 9.35, "Heart Disease Risk API: MLOps architecture", ha="center", fontsize=15, fontweight="bold"
    )

    # Lane 1: data and experimentation
    lane(ax, 6.55, 2.55, "1  DATA AND EXPERIMENTATION")
    box(ax, 0.5, 7.05, 2.6, 1.2, "UCI Heart Disease", "Cleveland, 303 rows, 14 columns", "data")
    box(ax, 3.9, 7.05, 3.3, 1.2, "Data pipeline", "download_data · data_processing · EDA", "data")
    box(ax, 8.0, 7.05, 3.3, 1.2, "train.py", "Pipeline + GridSearchCV (5-fold)", "ml")
    box(ax, 12.2, 7.55, 3.4, 0.95, "MLflow tracking", "params · metrics · plots · models", "ml")
    box(ax, 12.2, 6.65, 3.4, 0.75, "models/model.joblib", "preprocessing + best model", "ml")
    arrow(ax, (3.1, 7.65), (3.9, 7.65))
    arrow(ax, (7.2, 7.65), (8.0, 7.65), "clean CSV")
    arrow(ax, (11.3, 7.95), (12.2, 8.0), "log")
    arrow(ax, (11.3, 7.35), (12.2, 7.05), "save", dy=0.12)

    # Lane 2: CI/CD
    lane(ax, 3.7, 2.55, "2  CI/CD  (GitHub Actions)")
    box(ax, 0.5, 4.2, 2.3, 1.2, "GitHub repo", "code · tests · manifests", "ci")
    box(ax, 3.4, 4.2, 2.0, 1.2, "Lint", "flake8 + black", "ci")
    box(ax, 6.0, 4.2, 2.0, 1.2, "Unit tests", "pytest + coverage", "ci")
    box(ax, 8.6, 4.2, 2.3, 1.2, "Train", "train.py + MLflow summary", "ci")
    box(ax, 11.5, 4.2, 2.3, 1.2, "Docker build", "+ container smoke test", "ci")
    box(ax, 14.1, 4.2, 1.5, 1.2, "Artifacts", "model · logs", "ci")
    for a, b in [(2.8, 3.4), (5.4, 6.0), (8.0, 8.6), (10.9, 11.5), (13.8, 14.1)]:
        arrow(ax, (a, 4.8), (b, 4.8))
    ax.text(
        8,
        3.85,
        "pipeline fails fast: any lint, test, training or smoke-test error stops the run",
        ha="center",
        fontsize=8,
        style="italic",
        color="#7A5A12",
    )
    arrow(ax, (13.9, 6.65), (12.9, 5.4), "model", rad=-0.1)

    # Lane 3: serving and observability
    lane(ax, 0.2, 3.25, "3  SERVING AND OBSERVABILITY")
    box(ax, 0.5, 1.45, 2.3, 1.2, "Docker image", "FastAPI + model.joblib", "serve")
    box(ax, 3.6, 0.5, 4.3, 2.7, "", "", "serve")
    ax.text(5.75, 3.0, "Kubernetes (Minikube / Docker Desktop)", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 3.85, 1.0, 1.85, 1.2, "Deployment", "2 replicas · probes", "serve")
    box(ax, 5.85, 1.0, 1.85, 1.2, "Service", "LoadBalancer :80", "serve")
    box(ax, 8.9, 1.0, 1.6, 1.2, "Client", "curl / app", "serve")
    box(ax, 11.0, 2.05, 2.1, 1.0, "Prometheus", "scrapes /metrics", "obs")
    box(ax, 13.6, 2.05, 2.0, 1.0, "Grafana", "dashboard", "obs")
    box(ax, 11.0, 0.5, 4.6, 0.95, "Structured JSON request logs", "request_id · status · latency_ms", "obs")
    arrow(ax, (2.8, 1.7), (3.85, 1.6), "deploy")
    arrow(ax, (5.7, 1.6), (5.85, 1.6))
    arrow(ax, (8.9, 1.6), (7.7, 1.6), "POST /predict", dy=0.14)
    arrow(ax, (11.0, 2.55), (7.9, 2.55), "scrape /metrics", dy=0.14)
    arrow(ax, (13.1, 2.55), (13.6, 2.55))
    arrow(ax, (7.9, 0.75), (11.0, 0.75), "stdout logs", dy=0.13)
    # build artefact -> image, routed through the gap between lanes so it crosses no box
    elbow(ax, [(12.65, 4.2), (12.65, 3.58), (1.65, 3.58), (1.65, 2.65)])
    ax.text(
        7.0,
        3.36,
        "docker image",
        ha="center",
        fontsize=7.5,
        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "report" / "architecture.png"
    )
