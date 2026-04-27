"""Generate poster-ready visuals for phishing capstone results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
VISUALS = ROOT / "outputs" / "visuals"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _model_map(metrics_rows: List[Dict]) -> Dict[str, Dict]:
    return {str(r.get("model")): r for r in metrics_rows}


def dataset_composition() -> Path:
    merge = _read_json(REPORTS / "fresh_weighted_merge_summary.json")
    kaggle_rows = int(merge.get("kaggle_row_count", 0))
    fresh_rows = int(merge.get("fresh_row_count", 0))
    after = merge.get("class_distribution_after_merge", {})
    phish = int(after.get("1", 0))
    legit = int(after.get("0", 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].bar(["Kaggle", "Fresh"], [kaggle_rows, fresh_rows])
    axes[0].set_title("Dataset Source Rows", pad=10)
    axes[0].set_ylabel("Rows")
    axes[0].ticklabel_format(style="plain", axis="y")

    axes[1].bar(["Phishing", "Legitimate"], [phish, legit])
    axes[1].set_title("Class Distribution After Merge", pad=10)
    axes[1].set_ylabel("Rows")
    axes[1].ticklabel_format(style="plain", axis="y")

    fig.suptitle("Dataset Composition (Current Weighted Merge)", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = VISUALS / "dataset_composition.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out


def model_performance_standard() -> Path:
    std_0 = _model_map(_read_json(REPORTS / "exp_fresh_weight_0p0_standard_metrics.json"))
    std_25 = _model_map(_read_json(REPORTS / "exp_fresh_weight_0p25_standard_metrics.json"))
    models = ["logistic_regression", "random_forest", "xgboost"]
    labels = ["Logistic Regression", "Random Forest", "XGBoost"]

    f1_0 = [std_0[m]["f1"] for m in models]
    f1_25 = [std_25[m]["f1"] for m in models]
    auc_0 = [std_0[m]["roc_auc"] for m in models]
    auc_25 = [std_25[m]["roc_auc"] for m in models]

    x = np.arange(len(models))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(x - w / 2, f1_0, width=w, label="fresh_weight=0.0")
    axes[0].bar(x + w / 2, f1_25, width=w, label="fresh_weight=0.25")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=22, ha="right")
    axes[0].set_title("F1 Score (Standard Test)", pad=10)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right")
    axes[0].tick_params(axis="x", pad=8)

    axes[1].bar(x - w / 2, auc_0, width=w, label="fresh_weight=0.0")
    axes[1].bar(x + w / 2, auc_25, width=w, label="fresh_weight=0.25")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=22, ha="right")
    axes[1].set_title("ROC-AUC (Standard Test)", pad=10)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")
    axes[1].tick_params(axis="x", pad=8)

    fig.suptitle("Model Performance: Kaggle-only vs Kaggle+Fresh", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    out = VISUALS / "model_performance_standard.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out


def fresh_data_performance() -> Path:
    fresh_0 = _model_map(_read_json(REPORTS / "exp_fresh_weight_0p0_fresh_only_metrics.json")["metrics"])
    fresh_25 = _model_map(_read_json(REPORTS / "exp_fresh_weight_0p25_fresh_only_metrics.json")["metrics"])
    models = ["logistic_regression", "random_forest", "xgboost"]
    labels = ["Logistic Regression", "Random Forest", "XGBoost"]
    f1_0 = [fresh_0[m]["f1"] for m in models]
    f1_25 = [fresh_25[m]["f1"] for m in models]

    x = np.arange(len(models))
    w = 0.28
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, f1_0, width=w, label="fresh_weight=0.0")
    ax.bar(x + w / 2, f1_25, width=w, label="fresh_weight=0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("Fresh-only Evaluation (F1)", pad=12)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.06)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", pad=8)

    ymax = max(max(f1_0), max(f1_25))
    for i, (a, b) in enumerate(zip(f1_0, f1_25)):
        ax.text(i, ymax + 0.035, f"{b-a:+.3f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    out = VISUALS / "fresh_data_performance.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out


def model_selection_logic() -> Path:
    std_25 = _read_json(REPORTS / "exp_fresh_weight_0p25_standard_metrics.json")
    rows = [r for r in std_25 if r.get("model") in {"logistic_regression", "random_forest", "xgboost"}]
    rows.sort(key=lambda r: str(r["model"]))
    keys = [str(r["model"]) for r in rows]
    pretty = {
        "logistic_regression": "Logistic\nRegression",
        "random_forest": "Random\nForest",
        "xgboost": "XGBoost",
    }
    labels = [pretty.get(k, k) for k in keys]
    f1 = [float(r.get("f1", 0.0)) for r in rows]
    penalty = [float(r.get("audit_official_https_mean_phish_proba", 0.0)) for r in rows]
    composite = [a - b for a, b in zip(f1, penalty)]

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.bar(x - w, f1, width=w, label="F1")
    ax.bar(x, penalty, width=w, label="Official HTTPS false-positive signal")
    ax.bar(x + w, composite, width=w, label="Composite (F1 minus penalty)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Score")
    ax.set_title("Composite Selection Logic (Policy: composite)", pad=12)
    ax.set_ylim(0, max(1.05, max(f1) + 0.15, max(penalty) + 0.1, max(composite) + 0.25))
    ax.legend(loc="upper left", fontsize=11, ncol=1)
    ax.tick_params(axis="x", pad=10)

    best_idx = int(np.argmax(np.array(composite)))
    ax.text(
        0.5,
        0.98,
        f"Selected primary: {keys[best_idx].replace('_', ' ')}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = VISUALS / "model_selection_composite.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out


def pipeline_flow() -> Path:
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")

    boxes = {
        "kaggle": (0.06, 0.78, "Kaggle Dataset\n(~800k URLs)"),
        "clean": (0.26, 0.78, "Normalize + Clean\nCanonical dedupe"),
        "split": (0.46, 0.78, "Leak-safe Split\n(registered_domain)"),
        "train": (0.66, 0.78, "Train Models\nLR / RF / XGBoost"),
        "std_eval": (0.86, 0.78, "Standard Test\nMetrics"),
        "phishstats": (0.06, 0.42, "PhishStats\nRecent URLs"),
        "fresh_ds": (0.26, 0.42, "Fresh Dataset Build\n+ label mapping"),
        "fresh_merge": (0.46, 0.42, "Weighted Merge\n(--fresh-weight)"),
        "fresh_eval": (0.86, 0.42, "Fresh-only Eval\nReal-world validation"),
        "select": (0.66, 0.18, "Composite Selection\nF1 - HTTPS penalty"),
    }

    for x, y, text in boxes.values():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45"),
        )

    def arr(a: str, b: str):
        x1, y1, _ = boxes[a]
        x2, y2, _ = boxes[b]
        ax.annotate("", xy=(x2 - 0.07, y2), xytext=(x1 + 0.07, y1), arrowprops=dict(arrowstyle="->"))

    arr("kaggle", "clean")
    arr("clean", "split")
    arr("split", "train")
    arr("train", "std_eval")
    arr("phishstats", "fresh_ds")
    arr("fresh_ds", "fresh_merge")
    arr("fresh_merge", "train")
    arr("fresh_merge", "fresh_eval")
    arr("train", "select")

    ax.set_title("Phishing Detection Data/Training Pipeline", pad=16)
    out = VISUALS / "pipeline_flow.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    return out


def main() -> None:
    VISUALS.mkdir(parents=True, exist_ok=True)
    outputs = [
        dataset_composition(),
        model_performance_standard(),
        fresh_data_performance(),
        model_selection_logic(),
        pipeline_flow(),
    ]
    print("Generated visuals:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
