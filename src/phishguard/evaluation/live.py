"""`phishguard evaluate-live`: measure the full system with real Playwright page capture (rebuild Phase 4).

Needs internet access to arbitrary websites, so it runs in GitHub Actions
(.github/workflows/live-capture.yml), not in a locked-down sandbox.

For every URL the page is captured ONCE, then the full dashboard analysis runs twice on that same
capture: with the legitimacy-rescue layer on (normal) and off. That gives an apples-to-apples
ablation of the rescue layer without visiting any site twice.

URL sets
  eal_edge_cases     data/evaluation/eal_edge_cases.json (15 cases with expected outcomes)
  url_suites         data/evaluation/url_suites.json (18 curated URLs)
  hard_legit         data/evaluation/hard_legit_urls.jsonl (15 tricky legitimate URLs)
  official_brand     test half of data/evaluation/official_brand_urls.jsonl (136)
  phishstats_live    the newest phishing URLs from the PhishStats feed at run time (live, unseen)
  tranco_live        a seeded sample of popular legitimate homepages (pinned Tranco list)

Live sites change and phishing pages are taken down within hours, so capture failures are counted
and reported separately; a verdict on a dead page is not a detection.

Writes metrics/results/live_capture.json (+ re-renders metrics/VERIFIED_METRICS.md if evaluation.json exists).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import numpy as np

from phishguard.config import SEED
from phishguard.paths import project_root


def _load_sets(n_phishstats: int, n_tranco: int) -> Dict[str, List[Dict[str, Any]]]:
    from phishguard.data.eval_sets import (
        evaluation_exclusions, load_hard_legit_rows, load_official_brand_rows, load_url_suites,
    )

    root = project_root()
    edge = json.loads((root / "data" / "evaluation" / "eal_edge_cases.json").read_text())["cases"]
    sets: Dict[str, List[Dict[str, Any]]] = {
        "eal_edge_cases": [{"url": c["url"], "expected": c["expected"], "note": c.get("note")} for c in edge],
        "url_suites": [{"url": u, "expected": "not_phishing" if "legit" in k else "phishing", "bucket": k}
                       for k, v in load_url_suites().items() for u in v],
        "hard_legit": [{"url": r["url"], "expected": "not_phishing"} for r in load_hard_legit_rows()],
        "official_brand": [{"url": r["url"], "expected": "not_phishing"} for r in load_official_brand_rows(split="test")],
    }
    if n_phishstats:
        from phishguard.data.fresh_collect import collect_phishstats

        df = collect_phishstats(pages=max(1, n_phishstats // 100 + 1), max_rows=n_phishstats * 3)
        urls = []
        seen_hosts = set()
        for u in (df["url"].tolist() if len(df) and "url" in df.columns else []):
            h = str(u).split("//", 1)[-1].split("/", 1)[0].lower()
            if h and h not in seen_hosts:
                seen_hosts.add(h)
                urls.append(u)
            if len(urls) >= n_phishstats:
                break
        sets["phishstats_live"] = [{"url": u, "expected": "phishing"} for u in urls]
    if n_tranco:
        from phishguard.data.tranco import fetch_tranco
        from phishguard.urls.safe import leak_safe_group_key

        path = fetch_tranco()
        doms = [line.split(",", 1)[1].strip() for line in path.read_text().splitlines()[:5000] if "," in line]
        excl = evaluation_exclusions()["domains"]
        doms = [d for d in doms if leak_safe_group_key(d)[0] not in excl]
        random.Random(SEED).shuffle(doms)
        sets["tranco_live"] = [{"url": f"https://{d}/", "expected": "not_phishing"} for d in doms[:n_tranco]]
    return sets


def _passes(expected: str, verdict: str) -> bool:
    if expected == "phishing":
        return verdict == "likely_phishing"
    if expected == "phishing_or_uncertain":
        return verdict != "likely_legitimate"
    return verdict != "likely_phishing"  # not_phishing


def _analyze(url: str, capture, rescue: bool) -> Tuple[Dict[str, Any], float]:
    import phishguard.app.dashboard as dash

    os.environ["PHISH_LEGITIMACY_RESCUE_ENABLED"] = "true" if rescue else "false"
    t0 = time.perf_counter()
    with patch.object(dash, "capture_url", return_value=capture):
        payload, gaps = dash.build_dashboard_analysis(url, reinforcement=True)
    return payload, time.perf_counter() - t0


def run(n_phishstats: int = 60, n_tranco: int = 60, out: Path | None = None) -> Dict[str, Any]:
    from phishguard.app.capture import capture_url
    from phishguard.app.runtime_config import PipelineConfig
    from phishguard.evaluation.evaluate import provenance

    sets = _load_sets(n_phishstats, n_tranco)
    cfg = PipelineConfig.from_env()
    rows: List[Dict[str, Any]] = []
    for sname, items in sets.items():
        for it in items:
            url = it["url"]
            t0 = time.perf_counter()
            try:
                cap = capture_url(url, cfg, namespace="suspicious")
                cap_err = cap.error
            except Exception as e:  # noqa: BLE001
                cap, cap_err = None, f"{type(e).__name__}: {e}"
            t_cap = time.perf_counter() - t0
            rec = {"set": sname, **it, "capture_seconds": t_cap, "capture_error": cap_err,
                   "capture_strategy": getattr(cap, "capture_strategy", None),
                   "capture_ok": cap is not None and not cap_err and getattr(cap, "capture_strategy", "") != "failed"}
            if cap is not None:
                p_on, t_on = _analyze(url, cap, rescue=True)
                p_off, _ = _analyze(url, cap, rescue=False)
                v_on, v_off = p_on["verdict"]["verdict_3way"], p_off["verdict"]["verdict_3way"]
                rec.update({"verdict": v_on, "verdict_rescue_off": v_off, "analysis_seconds": t_on,
                            "rescue_applied": bool(p_on["verdict"].get("legitimacy_rescue_applied")),
                            "hard_blockers": p_on["verdict"].get("evidence_hard_blockers"),
                            "layer1_p": p_on["layer1_ml"].get("phish_proba"),
                            "passes": _passes(it["expected"], v_on)})
            rows.append(rec)
            print(f"[{sname}] {rec.get('verdict')} {url[:80]}", flush=True)
    os.environ["PHISH_LEGITIMACY_RESCUE_ENABLED"] = "true"

    def summarize(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        done = [r for r in rs if "verdict" in r]
        live = [r for r in done if r["capture_ok"]]
        v = [r["verdict"] for r in done]
        out = {"n": len(rs), "analyzed": len(done), "capture_ok": len(live),
               "capture_failure_rate": 1 - len(live) / len(rs) if rs else None,
               "verdicts": {k: v.count(k) for k in ("likely_phishing", "uncertain", "likely_legitimate")},
               "pass_rate": float(np.mean([r["passes"] for r in done])) if done else None,
               "pass_rate_capture_ok_only": float(np.mean([r["passes"] for r in live])) if live else None,
               "rescue_changed_verdicts": sum(r["verdict"] != r["verdict_rescue_off"] for r in done),
               "likely_phishing_rescue_on": sum(r["verdict"] == "likely_phishing" for r in done),
               "likely_phishing_rescue_off": sum(r["verdict_rescue_off"] == "likely_phishing" for r in done)}
        return out

    per_set = {s: summarize([r for r in rows if r["set"] == s]) for s in sets}
    tot = [r["capture_seconds"] + r.get("analysis_seconds", 0.0) for r in rows if "verdict" in r]
    report = {
        "provenance": provenance(),
        "command": "phishguard evaluate-live",
        "login_interaction": PipelineConfig.from_env().enable_login_interaction,
        "per_set": per_set,
        "edge_case_failures": [{k: r.get(k) for k in ("url", "expected", "verdict", "capture_ok", "note")}
                               for r in rows if r["set"] == "eal_edge_cases" and not r.get("passes")],
        "latency_seconds_full_path": {"n": len(tot), "p50": float(np.percentile(tot, 50)) if tot else None,
                                      "p95": float(np.percentile(tot, 95)) if tot else None,
                                      "note": "capture (Playwright) + full analysis, one URL at a time, GitHub-hosted runner"},
        "rows": rows,
    }
    out = out or (project_root() / "metrics" / "results" / "live_capture.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("wrote", out)
    ev = project_root() / "metrics" / "results" / "evaluation.json"
    if ev.is_file():
        from phishguard.evaluation.report import write_markdown

        write_markdown(json.loads(ev.read_text()), project_root() / "metrics" / "VERIFIED_METRICS.md",
                       project_root() / "docs" / "MODEL_CARD.md")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-system evaluation with live Playwright capture (needs internet).")
    ap.add_argument("--phishstats", type=int, default=60, help="newest PhishStats URLs to test (0 = skip)")
    ap.add_argument("--tranco", type=int, default=60, help="popular legitimate homepages to test (0 = skip)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run(a.phishstats, a.tranco, a.out)


if __name__ == "__main__":
    main()
