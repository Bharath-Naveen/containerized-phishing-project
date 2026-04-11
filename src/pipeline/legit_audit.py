"""Audit raw legit-*.txt files: counts, domain families, targets, coverage gaps."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List
from urllib.parse import urlparse

import tldextract

from src.pipeline.ingest import _read_txt_urls
from src.pipeline.paths import ensure_layout, raw_dir, reports_dir

logger = logging.getLogger(__name__)

LEGIT_PREFIX = "legit-"
TARGET_BRANDS = ("amazon", "google", "microsoft", "paypal")

# Minimum examples per brand per bucket for “adequate coverage” heuristics
MIN_LOGIN_AUTH = 6
MIN_RECOVERY_SUPPORT = 4  # recovery_verification + support_help
MIN_ACCOUNT_PRODUCT = 5  # account_dashboard + product_service + subdomain_product + checkout_payment


def _host_from_url(url: str) -> str:
    raw = (url or "").strip()
    p = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (p.netloc or "").lower()
    if "@" in host:
        host = host.split("@")[-1]
    if ":" in host and not host.startswith("["):
        h, port = host.rsplit(":", 1)
        if port.isdigit():
            host = h
    return host


def domain_subdomain_family(url: str) -> str:
    """Bucket: first subdomain label + registered domain (e.g. accounts|google.com)."""
    host = _host_from_url(url)
    if not host:
        return "unknown"
    ext = tldextract.extract(host)
    reg = ".".join(p for p in (ext.domain, ext.suffix) if p)
    if not reg:
        return host
    if host == reg or host.endswith("." + reg):
        prefix = host[: -(len(reg) + 1)] if host.endswith("." + reg) else ""
        first = prefix.split(".")[0] if prefix else "@apex"
        if not first:
            first = "@apex"
        return f"{first}|{reg}"
    return f"@other|{reg}"


def _load_legit_rows(raw_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(raw_root.glob(f"{LEGIT_PREFIX}*.txt")):
        brand = path.name.lower()[len(LEGIT_PREFIX) : -4]
        for cat, url in _read_txt_urls(path):
            if not url:
                continue
            rows.append(
                {
                    "brand": brand,
                    "action_category": cat or "uncategorized",
                    "url": url,
                    "source_file": path.name,
                }
            )
    return rows


def _recommend_targets(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    n = len(rows)
    n_brands = len(TARGET_BRANDS)
    per_brand_floor = max(8, int(math.ceil(n / max(n_brands, 1) * 0.85)))
    per_brand_ceiling = int(math.ceil(n / max(n_brands, 1) * 1.25)) if n else 0

    min_per_major_cat = max(5, int(round(math.sqrt(max(n, 1)))))
    major = (
        "login_auth",
        "homepage",
        "account_dashboard",
        "recovery_verification",
        "support_help",
        "product_service",
        "checkout_payment",
        "subdomain_product",
    )
    per_category_targets = {c: min_per_major_cat for c in major}
    per_category_targets["homepage"] = max(3, min_per_major_cat // 2)

    return {
        "rationale": (
            "Per-brand floor/ceiling keeps legit diversity (~equal weight per target brand). "
            "Per-category floors push coverage of login, recovery, account, and product surfaces."
        ),
        "per_brand": {
            "target_floor": per_brand_floor,
            "target_ceiling": per_brand_ceiling,
            "note": "Aim for each of amazon/google/microsoft/paypal within [floor, ceiling] rows.",
        },
        "per_action_category": {
            "min_examples_for_major_categories": per_category_targets,
            "note": "Increase URLs in under-filled categories until counts meet or exceed these mins.",
        },
        "totals_reference": {"n_legit_urls": n, "n_brands": n_brands},
    }


def _coverage_gaps(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    by_brand: DefaultDict[str, List[str]] = defaultdict(list)
    for r in rows:
        by_brand[r["brand"].lower()].append(r["action_category"].lower())

    gaps: Dict[str, Any] = {"by_brand": {}}
    for b in TARGET_BRANDS:
        cats = by_brand.get(b, [])
        n_login = sum(1 for c in cats if c == "login_auth")
        n_rec = sum(1 for c in cats if c in ("recovery_verification", "support_help"))
        n_acct_prod = sum(
            1
            for c in cats
            if c
            in (
                "account_dashboard",
                "product_service",
                "subdomain_product",
                "checkout_payment",
            )
        )
        entry = {
            "n_rows": len(cats),
            "login_auth": n_login,
            "recovery_support": n_rec,
            "account_or_product_surfaces": n_acct_prod,
            "needs_more_login_auth": n_login < MIN_LOGIN_AUTH,
            "needs_more_recovery_support": n_rec < MIN_RECOVERY_SUPPORT,
            "needs_more_account_or_product": n_acct_prod < MIN_ACCOUNT_PRODUCT,
        }
        gaps["by_brand"][b] = entry
    gaps["thresholds"] = {
        "min_login_auth": MIN_LOGIN_AUTH,
        "min_recovery_support": MIN_RECOVERY_SUPPORT,
        "min_account_or_product": MIN_ACCOUNT_PRODUCT,
    }
    return gaps


def audit_legit_raw_files(raw_root: Path | None = None) -> Dict[str, Any]:
    ensure_layout()
    root = raw_root or raw_dir()
    rows = _load_legit_rows(root)
    if not rows:
        out = {
            "error": "no_legit_txt_rows",
            "raw_dir": str(root),
            "counts_by_brand": {},
            "counts_by_action_category": {},
            "counts_by_domain_family": {},
        }
        (reports_dir() / "legit_raw_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    by_brand: Counter = Counter()
    by_cat: Counter = Counter()
    by_family: Counter = Counter()
    for r in rows:
        by_brand[r["brand"].lower()] += 1
        by_cat[r["action_category"].lower()] += 1
        by_family[domain_subdomain_family(r["url"])] += 1

    targets = _recommend_targets(rows)
    coverage = _coverage_gaps(rows)

    report: Dict[str, Any] = {
        "raw_dir": str(root),
        "n_lines": len(rows),
        "counts_by_brand": dict(sorted(by_brand.items())),
        "counts_by_action_category": dict(sorted(by_cat.items())),
        "counts_by_domain_subdomain_family": dict(sorted(by_family.items())),
        "target_legit_counts_recommendation": targets,
        "coverage_gaps": coverage,
    }
    path = reports_dir() / "legit_raw_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = reports_dir() / "legit_raw_audit.md"
    lines = [
        "# Legit raw file audit\n",
        f"- **Lines (URLs):** {len(rows)}\n",
        "\n## By brand\n",
    ]
    for k, v in sorted(by_brand.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## By action_category\n")
    for k, v in sorted(by_cat.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## By domain / subdomain family\n")
    for k, v in sorted(by_family.items(), key=lambda x: -x[1])[:60]:
        lines.append(f"- `{k}`: {v}\n")
    if len(by_family) > 60:
        lines.append(f"- _… {len(by_family) - 60} more families_\n")
    lines.append("\n## Recommended targets\n")
    lines.append(
        f"- Per brand floor: **{targets['per_brand']['target_floor']}**, "
        f"ceiling: **{targets['per_brand']['target_ceiling']}**\n"
    )
    lines.append("\n## Coverage gaps (heuristic)\n")
    for b, g in coverage["by_brand"].items():
        flags = []
        if g["needs_more_login_auth"]:
            flags.append("login/auth")
        if g["needs_more_recovery_support"]:
            flags.append("recovery/support")
        if g["needs_more_account_or_product"]:
            flags.append("account/dashboard or product/service")
        if flags:
            lines.append(f"- **{b}**: needs more — {', '.join(flags)}\n")
        else:
            lines.append(f"- **{b}:** OK vs thresholds\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote %s and %s", path, md_path)
    return report


def main() -> None:
    from src.pipeline.logging_util import setup_logging
    from src.pipeline.paths import logs_dir

    setup_logging(logs_dir() / "legit_audit.log")
    p = argparse.ArgumentParser(description="Audit legit-*.txt raw files.")
    p.add_argument("--raw-dir", type=Path, default=None)
    args = p.parse_args()
    audit_legit_raw_files(args.raw_dir)


if __name__ == "__main__":
    main()
