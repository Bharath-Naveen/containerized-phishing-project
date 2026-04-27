"""Poster visuals for the deployed layer1_primary model only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics" / "metrics.json"
REPORTS = ROOT / "outputs" / "reports"
VISUALS = ROOT / "outputs" / "visuals"

PRIMARY = MODELS / "layer1_primary.joblib"
CANDIDATES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")
EXP_0 = REPORTS / "exp_fresh_weight_0p0_standard_metrics.json"
EXP_25 = REPORTS / "exp_fresh_weight_0p25_standard_metrics.json"

DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def _poster_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
        }
    )


def identify_deployed_model_key() -> str:
    if not PRIMARY.is_file():
        raise FileNotFoundError(f"Missing deployed model: {PRIMARY}")
    primary_bytes = PRIMARY.read_bytes()
    for key in CANDIDATES:
        p = MODELS / f"{key}.joblib"
        if p.is_file() and p.read_bytes() == primary_bytes:
            return key
    raise RuntimeError(
        "Could not match layer1_primary.joblib to a known candidate "
        f"({', '.join(CANDIDATES)}). Re-run training or check outputs/models/."
    )


def _metrics_row(rows: List[Dict], model_key: str) -> Dict:
    for r in rows:
        if str(r.get("model")) == model_key:
            return r
    raise KeyError(f"No metrics row for model={model_key}")


def _metric_series(row: Dict) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    values: List[float] = []
    pairs = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]
    for lab, key in pairs:
        v = row.get(key)
        if v is None and key == "roc_auc":
            continue
        if v is None:
            continue
        labels.append(lab)
        values.append(float(v))
    return labels, values


def chart_deployed_metrics(model_key: str, row: Dict) -> Path:
    labels, values = _metric_series(row)
    display = DISPLAY.get(model_key, model_key)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Deployed Model Performance", pad=10)
    ax.text(
        0.5,
        1.045,
        f"Model: {display}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
    )
    ax.tick_params(axis="x", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = VISUALS / "deployed_model_metrics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)
    return out


def chart_baseline_vs_fresh(model_key: str) -> Optional[Path]:
    if not EXP_0.is_file() or not EXP_25.is_file():
        return None
    rows0 = json.loads(EXP_0.read_text(encoding="utf-8"))
    rows25 = json.loads(EXP_25.read_text(encoding="utf-8"))
    r0 = _metrics_row(rows0, model_key)
    r25 = _metrics_row(rows25, model_key)

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    v0: List[float] = []
    v25: List[float] = []
    labels: List[str] = []
    for lab, k in zip(metric_labels, metric_keys):
        a0, a25 = r0.get(k), r25.get(k)
        if a0 is None and a25 is None:
            continue
        if k == "roc_auc" and (a0 is None or a25 is None):
            continue
        labels.append(lab)
        v0.append(float(a0) if a0 is not None else float("nan"))
        v25.append(float(a25) if a25 is not None else float("nan"))

    if not labels:
        return None

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.bar(x - w / 2, v0, width=w, label="fresh_weight = 0.0")
    ax.bar(x + w / 2, v25, width=w, label="fresh_weight = 0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    display = DISPLAY.get(model_key, model_key)
    ax.set_title(f"Deployed Model ({display}): Training Mix Comparison", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.tick_params(axis="x", pad=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = VISUALS / "deployed_model_baseline_vs_fresh.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)
    return out


def main() -> None:
    _poster_rc()
    VISUALS.mkdir(parents=True, exist_ok=True)

    model_key = identify_deployed_model_key()
    rows = json.loads(METRICS.read_text(encoding="utf-8"))
    row = _metrics_row(rows, model_key)

    p1 = chart_deployed_metrics(model_key, row)
    print(p1)
    print(
        "  Caption: Bar chart of accuracy, precision, recall, F1, and ROC-AUC for the "
        f"deployed dashboard model ({DISPLAY.get(model_key, model_key)}), from the latest standard test evaluation."
    )

    p2 = chart_baseline_vs_fresh(model_key)
    if p2 is not None:
        print(p2)
        print(
            "  Caption: Same deployed model type compared on the standard test set after training with "
            "fresh_weight=0.0 versus fresh_weight=0.25 (when experiment snapshots are present)."
        )
    else:
        print("(skipped) outputs/visuals/deployed_model_baseline_vs_fresh.png — missing exp_fresh_weight_*_standard_metrics.json")


if __name__ == "__main__":
    main()
