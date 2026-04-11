"""Summarize dataset composition: brand, label, action categories, train balance."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.pipeline.legit_audit import audit_legit_raw_files
from src.pipeline.paths import ensure_layout, interim_dir, processed_dir, reports_dir

logger = logging.getLogger(__name__)

LOGIN_LIKE = frozenset({"login_auth", "recovery_verification", "account_dashboard"})
HOMEPAGE_LIKE = frozenset({"homepage"})


def _entropy(counts: List[int]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / tot
        h -= p * math.log(p + 1e-12, 2)
    return round(h, 4)


def _summarize_frame(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    if df.empty:
        return {"name": name, "n_rows": 0}
    lab = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
    brand = df.get("source_brand_hint", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    cat = df.get("action_category", pd.Series(["uncategorized"] * len(df))).astype(str).str.lower()

    n = len(df)
    n0 = int((lab == 0).sum())
    n1 = int((lab == 1).sum())
    by_brand = brand.value_counts().to_dict()
    by_label = {"legit_0": n0, "phish_1": n1}
    by_cat = cat.value_counts().to_dict()

    legit_mask = lab == 0
    phish_mask = lab == 1
    legit_cats = cat[legit_mask]
    phish_cats = cat[phish_mask]

    legit_home = int(legit_cats.isin(HOMEPAGE_LIKE).sum())
    legit_login = int(legit_cats.isin(LOGIN_LIKE).sum())
    legit_other = int(legit_mask.sum()) - legit_home - legit_login

    diversity = {
        "action_category_entropy_all": _entropy(list(by_cat.values())),
        "action_category_entropy_legit": _entropy(legit_cats.value_counts().tolist())
        if legit_mask.any()
        else 0.0,
        "action_category_entropy_phish": _entropy(phish_cats.value_counts().tolist())
        if phish_mask.any()
        else 0.0,
        "legit_unique_categories": int(legit_cats.nunique()) if legit_mask.any() else 0,
        "phish_unique_categories": int(phish_cats.nunique()) if phish_mask.any() else 0,
    }

    legit_non_home_share = round((n0 - legit_home) / max(n0, 1), 4) if n0 else 0.0
    login_flow_share_legit = round(legit_login / max(n0, 1), 4) if n0 else 0.0

    checks = {
        "legit_homepage_share": round(legit_home / max(n0, 1), 4) if n0 else 0.0,
        "legit_login_flow_share": login_flow_share_legit,
        "legit_not_only_homepage": legit_non_home_share >= 0.35,
        "login_flows_adequate": login_flow_share_legit >= 0.12,
        "legit_more_diverse_than_phish_categories": diversity["legit_unique_categories"]
        >= diversity["phish_unique_categories"],
    }

    return {
        "name": name,
        "n_rows": n,
        "by_label": by_label,
        "by_brand": {str(k): int(v) for k, v in by_brand.items()},
        "by_action_category": {str(k): int(v) for k, v in by_cat.items()},
        "legit_surface_mix": {
            "homepage_rows": legit_home,
            "login_flow_rows": legit_login,
            "other_rows": max(0, legit_other),
        },
        "diversity": diversity,
        "quality_checks": checks,
    }


def build_full_report(
    *,
    normalized: Optional[Path] = None,
    cleaned: Optional[Path] = None,
    train: Optional[Path] = None,
    test: Optional[Path] = None,
) -> Dict[str, Any]:
    ensure_layout()
    paths = {
        "normalized": normalized or (interim_dir() / "normalized.csv"),
        "cleaned": cleaned or (processed_dir() / "cleaned.csv"),
        "train": train or (processed_dir() / "train.csv"),
        "test": test or (processed_dir() / "test.csv"),
    }
    report: Dict[str, Any] = {"sources": {k: str(v) for k, v in paths.items()}, "sections": []}

    for key, p in paths.items():
        if not p.exists():
            report["sections"].append({"name": key, "error": "file_missing", "path": str(p)})
            continue
        df = pd.read_csv(p, dtype=str, low_memory=False)
        if "action_category" not in df.columns:
            df["action_category"] = "uncategorized"
        report["sections"].append(_summarize_frame(df, key))

    # Narrative comparison: normalized legit vs phish
    norm = paths["normalized"]
    narrative: List[str] = []
    if norm.exists():
        df = pd.read_csv(norm, dtype=str, low_memory=False)
        if "action_category" not in df.columns:
            df["action_category"] = "uncategorized"
        lab = pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int)
        legit = df[lab == 0]
        phish = df[lab == 1]
        lc = legit["action_category"].astype(str).str.lower()
        narrative.append(
            f"Normalized: {len(legit)} legit rows across {lc.nunique()} action categories; "
            f"{len(phish)} phish rows across {phish['action_category'].nunique()} categories."
        )
        if len(legit):
            lh = int(lc.isin(HOMEPAGE_LIKE).sum())
            ll = int(lc.isin(LOGIN_LIKE).sum())
            narrative.append(
                f"Legit mix: homepage≈{lh} ({lh/len(legit):.1%}), login/recovery/account≈{ll} ({ll/len(legit):.1%})."
            )

    report["narrative"] = narrative

    try:
        report["legit_raw_audit"] = audit_legit_raw_files()
    except Exception as e:
        logger.warning("legit_raw_audit failed: %s", e)
        report["legit_raw_audit"] = {"error": str(e)}

    balance_block: Dict[str, Any] = {}
    ml_stats_p = reports_dir() / "ml_prepare_stats.json"
    if ml_stats_p.exists():
        try:
            mls = json.loads(ml_stats_p.read_text(encoding="utf-8"))
            balance_block["after_prepare_ml_dataset"] = mls.get("balance_after_prepare_ml")
            balance_block["ml_prepare_stats_path"] = str(ml_stats_p)
        except Exception as e:
            balance_block["ml_prepare_stats_error"] = str(e)
    split_bal_p = reports_dir() / "split_balance_report.json"
    if split_bal_p.exists():
        try:
            balance_block["after_split_and_optional_train_balance"] = json.loads(
                split_bal_p.read_text(encoding="utf-8")
            )
        except Exception as e:
            balance_block["split_balance_error"] = str(e)
    tb_p = reports_dir() / "train_balance_stats.json"
    if tb_p.exists():
        try:
            balance_block["train_balance_job"] = json.loads(tb_p.read_text(encoding="utf-8"))
        except Exception as e:
            balance_block["train_balance_error"] = str(e)
    report["training_balance_reports"] = balance_block

    if norm.exists():
        ndf = pd.read_csv(norm, dtype=str, low_memory=False)
        if "action_category" not in ndf.columns:
            ndf["action_category"] = "uncategorized"
        nlab = pd.to_numeric(ndf["label"], errors="coerce").fillna(1).astype(int)
        legit = ndf[nlab == 0]
        phish = ndf[nlab == 1]
        lc = legit["action_category"].astype(str).str.lower()
        pc = phish["action_category"].astype(str).str.lower()
        report["comparison_legit_vs_phish"] = {
            "legit_rows": len(legit),
            "phish_rows": len(phish),
            "legit_action_categories_observed": int(lc.nunique()),
            "phish_action_categories_observed": int(pc.nunique()),
            "phish_uncategorized_fraction": round(float((pc == "uncategorized").mean()), 4)
            if len(phish)
            else 0.0,
            "legit_dominates_uncategorized": bool((lc == "uncategorized").mean() < 0.05)
            if len(legit)
            else False,
            "interpretation": (
                "Legit set uses structured action categories; phishing feed is mostly uncategorized — "
                "expected for legacy phish .txt lines. Compare HTML/URL features at train time, not category tags."
            ),
        }

    out_json = reports_dir() / "dataset_strategy_report.json"
    out_md = reports_dir() / "dataset_strategy_report.md"
    reports_dir().mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = ["# Dataset strategy report", ""]
    for sec in report["sections"]:
        if "error" in sec:
            md_lines.append(f"## {sec['name']}\n\n_Missing: {sec['path']}_\n")
            continue
        md_lines.append(f"## {sec['name']} ({sec['n_rows']} rows)\n")
        bl = sec.get("by_label", {})
        md_lines.append(
            f"- **Labels:** legit={bl.get('legit_0', 0)} phish={bl.get('phish_1', 0)}\n"
        )
        bb = sec.get("by_brand", {})
        if bb:
            md_lines.append(
                "- **By brand:** " + ", ".join(f"{k}={v}" for k, v in sorted(bb.items())) + "\n"
            )
        qc = sec.get("quality_checks", {})
        if qc:
            md_lines.append("- **Quality:** " + "; ".join(f"{k}={v}" for k, v in qc.items()) + "\n")
        sm = sec.get("legit_surface_mix", {})
        if sm:
            md_lines.append(
                f"- **Legit surfaces:** homepage={sm.get('homepage_rows')}, "
                f"login_flows={sm.get('login_flow_rows')}, other={sm.get('other_rows')}\n"
            )
        md_lines.append("")
    cmp_ = report.get("comparison_legit_vs_phish")
    if cmp_:
        md_lines.append("## Legit vs phishing (normalized)\n")
        for k, v in cmp_.items():
            md_lines.append(f"- **{k}:** {v}\n")
        md_lines.append("")
    md_lines.append("## Notes\n")
    for line in report.get("narrative", []):
        md_lines.append(f"- {line}\n")

    bal = report.get("training_balance_reports") or {}
    apm = bal.get("after_prepare_ml_dataset")
    if apm:
        md_lines.append("\n## Training balance (after prepare_ml_dataset)\n")
        for k, v in apm.items():
            md_lines.append(f"- **{k}:** {v}\n")
    asb = bal.get("after_split_and_optional_train_balance") or {}
    if asb:
        md_lines.append("\n## Training balance (split / train downsample)\n")
        b4 = asb.get("after_split_before_train_balance") or {}
        tr0 = (b4.get("train") or {})
        te0 = (b4.get("test") or {})
        md_lines.append(
            f"- **Train (after split):** legit={tr0.get('legit_0')} phish={tr0.get('phish_1')} "
            f"ratio={tr0.get('ratio_phish_per_legit')}\n"
        )
        md_lines.append(
            f"- **Test:** legit={te0.get('legit_0')} phish={te0.get('phish_1')} "
            f"ratio={te0.get('ratio_phish_per_legit')}\n"
        )
        aft = asb.get("after_train_phish_downsample") or {}
        if aft.get("train"):
            tr1 = aft["train"]
            md_lines.append(
                f"- **Train (after phish downsample):** legit={tr1.get('legit_0')} phish={tr1.get('phish_1')} "
                f"ratio={tr1.get('ratio_phish_per_legit')}\n"
            )
    lra = report.get("legit_raw_audit") or {}
    if lra and "error" not in lra and "counts_by_brand" in lra:
        md_lines.append("\n## Legit raw audit (summary)\n")
        md_lines.append(f"- **URLs:** {lra.get('n_lines', 0)}\n")
        bb = lra.get("counts_by_brand") or {}
        if bb:
            md_lines.append(
                "- **By brand:** " + ", ".join(f"{k}={v}" for k, v in sorted(bb.items())) + "\n"
            )
        cg = (lra.get("coverage_gaps") or {}).get("by_brand") or {}
        if cg:
            md_lines.append("- **Coverage gaps:**\n")
            for b, g in cg.items():
                need = []
                if g.get("needs_more_login_auth"):
                    need.append("login/auth")
                if g.get("needs_more_recovery_support"):
                    need.append("recovery/support")
                if g.get("needs_more_account_or_product"):
                    need.append("account/product")
                if need:
                    md_lines.append(f"  - {b}: needs {', '.join(need)}\n")

    out_md.write_text("".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s and %s", out_json, out_md)
    return report


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "dataset_report.log")
    p = argparse.ArgumentParser(description="Dataset composition report.")
    p.add_argument("--normalized", type=Path, default=None)
    p.add_argument("--cleaned", type=Path, default=None)
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--test", type=Path, default=None)
    args = p.parse_args()
    build_full_report(
        normalized=args.normalized,
        cleaned=args.cleaned,
        train=args.train,
        test=args.test,
    )


if __name__ == "__main__":
    main()
