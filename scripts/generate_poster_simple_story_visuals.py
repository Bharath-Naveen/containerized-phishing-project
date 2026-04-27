"""Poster-ready visuals: simple system story (no dataset-source detail)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics" / "metrics.json"
VISUALS = ROOT / "outputs" / "visuals"


def _poster_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
        }
    )


def _read_metrics() -> List[Dict]:
    if not METRICS.is_file():
        raise FileNotFoundError(f"Missing metrics file: {METRICS}")
    return json.loads(METRICS.read_text(encoding="utf-8"))


def _box(
    ax,
    xy: Tuple[float, float],
    text: str,
    width: float = 0.22,
    height: float = 0.12,
    *,
    fontsize: int = 11,
) -> None:
    x, y = xy
    ax.add_patch(
        plt.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            fill=True,
            linewidth=1.5,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def system_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Phishing Detection System Pipeline", pad=16)

    steps = [
        (0.10, 0.58, "User enters\nURL"),
        (0.30, 0.58, "URL capture /\nHTML extraction"),
        (0.50, 0.58, "Feature\nextraction"),
        (0.70, 0.58, "ML model\nprediction"),
        (0.90, 0.58, "Dashboard\nverdict"),
    ]
    w_box, h_box = 0.14, 0.34
    for x, y, t in steps:
        _box(ax, (x, y), t, width=w_box, height=h_box, fontsize=11)

    for i in range(len(steps) - 1):
        x0, y0, _ = steps[i]
        x1, y1, _ = steps[i + 1]
        ax.annotate(
            "",
            xy=(x1 - w_box / 2 - 0.01, y1),
            xytext=(x0 + w_box / 2 + 0.01, y0),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    ax.text(
        0.5,
        0.12,
        "Explanation: key features, scores, and capture artifacts are shown to the user in the dashboard.",
        ha="center",
        va="top",
        fontsize=12,
        linespacing=1.35,
        wrap=True,
    )

    out = VISUALS / "system_pipeline.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def feature_layers() -> Path:
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Feature Extraction (Major Groups)", pad=14)

    layers = [
        (0.5, 0.88, "URL structure\n(path, query, tokens)"),
        (0.5, 0.70, "Domain & redirect\n(host, registrable domain, redirects)"),
        (0.5, 0.52, "HTML & login surface\n(forms, password fields, DOM cues)"),
        (0.5, 0.34, "Brand / task inference\n(heuristics + structured hints)"),
        (0.5, 0.16, "Capture metadata\n(load outcome, strategy, timing)"),
    ]
    for x, y, t in layers:
        _box(ax, (x, y), t, width=0.70, height=0.11, fontsize=11)

    for i in range(len(layers) - 1):
        _, y0, _ = layers[i]
        _, y1, _ = layers[i + 1]
        ax.annotate("", xy=(0.5, y1 + 0.058), xytext=(0.5, y0 - 0.058), arrowprops=dict(arrowstyle="->", lw=2))

    ax.text(
        0.5,
        0.035,
        "Features feed the ML layer and dashboard explanations (no single signal decides alone).",
        ha="center",
        va="top",
        fontsize=12,
        linespacing=1.35,
    )

    out = VISUALS / "feature_layers.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def model_comparison() -> Path:
    rows = _read_metrics()
    order = ["logistic_regression", "random_forest", "xgboost"]
    by_model: Dict[str, Dict] = {str(r["model"]): r for r in rows}
    labels = ["Logistic Regression", "Random Forest", "XGBoost"]
    f1 = [float(by_model[m]["f1"]) for m in order]

    x = np.arange(len(order))
    w = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x, f1, width=w)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=22, ha="right")
    axes[0].set_ylabel("F1 score")
    axes[0].set_title("F1 (held-out test)")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", pad=8)

    roc_vals = [float(by_model[m]["roc_auc"]) if by_model[m].get("roc_auc") is not None else 0.0 for m in order]
    axes[1].bar(x, roc_vals, width=w)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=22, ha="right")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("ROC-AUC (held-out test)")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", pad=8)

    fig.suptitle("Model Comparison (Layer-1 URL/host features)", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    out = VISUALS / "model_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out


def dashboard_layers() -> Path:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dashboard Layers (What the User Sees)", pad=14)

    items = [
        (0.5, 0.90, "Verdict summary\n(risk label + confidence)"),
        (0.5, 0.72, "Capture provenance\n(status, strategy, limitations)"),
        (0.5, 0.54, "URL / domain signals\n(anchors, redirects, host cues)"),
        (0.5, 0.36, "HTML / content signals\n(forms, auth-like patterns)"),
        (0.5, 0.18, "Model explanation\n(key drivers, scores)"),
        (0.5, 0.04, "Artifacts\n(screenshot / HTML when available)"),
    ]
    for x, y, t in items:
        _box(ax, (x, y), t, width=0.74, height=0.10, fontsize=11)

    for i in range(len(items) - 1):
        _, y0, _ = items[i]
        _, y1, _ = items[i + 1]
        ax.annotate("", xy=(0.5, y1 + 0.052), xytext=(0.5, y0 - 0.052), arrowprops=dict(arrowstyle="->", lw=2))

    out = VISUALS / "dashboard_layers.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def model_selection_logic() -> Path:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Why One Model Is Selected for the Demo", pad=14)

    pillars = [
        (0.2, 0.56, "Predictive\nperformance\n(F1, ROC-AUC)"),
        (0.5, 0.56, "Fewer embarrassing\nfalse positives on\nofficial HTTPS sites"),
        (0.8, 0.56, "Stable, interpretable\nbehavior for\nshowcase / demo"),
    ]
    for x, y, t in pillars:
        _box(ax, (x, y), t, width=0.24, height=0.36, fontsize=11)

    ax.text(
        0.5,
        0.14,
        "Selection balances raw test metrics with audit behavior on high-trust reference URLs.",
        ha="center",
        va="top",
        fontsize=13,
        linespacing=1.35,
    )

    out = VISUALS / "model_selection_logic.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def main() -> None:
    _poster_rc()
    VISUALS.mkdir(parents=True, exist_ok=True)
    paths = [
        system_pipeline(),
        feature_layers(),
        model_comparison(),
        dashboard_layers(),
        model_selection_logic(),
    ]
    captions = [
        "End-to-end flow from a submitted URL through capture, features, ML, dashboard verdict, and user-facing explanation.",
        "Major feature groups extracted from the URL, page fetch, HTML, and structured heuristics before model scoring.",
        "Side-by-side F1 and ROC-AUC on a held-out test set for logistic regression, random forest, and XGBoost.",
        "Layered dashboard presentation from high-level verdict down to evidence, model rationale, and optional artifacts.",
        "Primary model choice trades off predictive strength, conservative behavior on official HTTPS references, and demo stability.",
    ]
    print("Generated:")
    for p, cap in zip(paths, captions):
        print(p)
        print("  Caption:", cap)


if __name__ == "__main__":
    main()
