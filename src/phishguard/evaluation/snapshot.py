"""Frozen live-capture snapshot: capture once, replay many times (rule tuning, docs/rebuild/TUNING_PLAN.md).

`phishguard snapshot-live` (GitHub Actions, needs internet) captures every URL once with Playwright and
writes one gzipped JSON-lines file: the capture record plus the page HTML for each URL. Screenshots are
not kept (the analysis does not read them).

`phishguard replay --split val|test|all` feeds those saved captures back into the full dashboard
analysis. Once it has a capture the analysis makes no network calls, so a replay gives the same
verdicts every time for the same code. That is what makes before/after rule comparisons fair.

Every `--split test` replay is appended to metrics/results/test_access_log.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from phishguard.config import SEED
from phishguard.paths import project_root

SNAPSHOT_DIR = Path("data") / "evaluation" / "frozen"
MAX_HTML_BYTES = 1_500_000
LEGIT_SETS = {"official_brand", "tranco", "hard_legit", "url_suites_legit"}


def hash_split(registered_domain: str) -> str:
    """Deterministic val/test split by registered domain: even first sha256 byte = val."""
    return "val" if hashlib.sha256(registered_domain.encode("utf-8")).digest()[0] % 2 == 0 else "test"


def _reg(url: str) -> str:
    from phishguard.urls.safe import leak_safe_group_key

    return leak_safe_group_key(url)[0]


def _host(url: str) -> str:
    from phishguard.urls.safe import safe_hostname

    h = (safe_hostname(url)[0] or "").lower().rstrip(".")
    return h[4:] if h.startswith("www.") else h


def _tranco_domains(top: int) -> List[str]:
    from phishguard.data.tranco import fetch_tranco

    lines = fetch_tranco().read_text().splitlines()[:top]
    return [ln.split(",", 1)[1].strip() for ln in lines if "," in ln]


def _items(n_phish: int, n_tranco: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from phishguard.data.eval_sets import (
        evaluation_exclusions, load_hard_legit_rows, load_official_brand_rows, load_url_suites,
    )
    from phishguard.evaluation.live import _fresh_phishing_urls

    root = project_root()
    meta: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    for r in load_official_brand_rows():
        items.append({"set": "official_brand", "url": r["url"], "expected": "not_phishing", "split": r["split"]})
    for r in load_hard_legit_rows():
        items.append({"set": "hard_legit", "url": r["url"], "expected": "not_phishing", "split": "val"})
    for k, urls in load_url_suites().items():
        for u in urls:
            legit = "legit" in k
            items.append({"set": "url_suites_legit" if legit else "url_suites_phish", "url": u, "bucket": k,
                          "expected": "not_phishing" if legit else "phishing", "split": "val"})
    edge = json.loads((root / "data" / "evaluation" / "eal_edge_cases.json").read_text())["cases"]
    for c in edge:
        items.append({"set": "eal_edge_cases", "url": c["url"], "expected": c["expected"], "note": c.get("note"),
                      "split": "val"})

    # Fresh phishing: one URL per host, split by host hash. Host, not registered domain, because much
    # phishing sits on user subdomains of big platforms (x.godaddysites.com, y.weebly.com) and grouping
    # by registered domain would collapse all of those into one URL.
    top10k = set(_tranco_domains(10_000))
    cands, meta["phishing_feed"] = _fresh_phishing_urls(n_phish * 4)
    seen = set()
    for u in cands:
        h = _host(u)
        if not h or h in seen:
            continue
        seen.add(h)
        items.append({"set": "fresh_phishing", "url": u, "expected": "phishing", "split": hash_split(h),
                      "host": h, "registered_domain": _reg(u), "popular_domain": h in top10k})
        if len(seen) >= n_phish:
            break

    # Popular homepages: pinned Tranco list, seeded shuffle, resolving domains only.
    excl = evaluation_exclusions()["domains"]
    doms = [d for d in _tranco_domains(5000) if _reg(d) not in excl]
    random.Random(SEED).shuffle(doms)
    kept, skipped = 0, 0
    for d in doms:
        if kept >= n_tranco:
            break
        try:
            socket.getaddrinfo(d, 443)
        except OSError:
            skipped += 1
            continue
        items.append({"set": "tranco", "url": f"https://{d}/", "expected": "not_phishing",
                      "split": hash_split(_reg(d)), "registered_domain": _reg(d)})
        kept += 1
    meta["tranco"] = {"list_id": "K9QPW", "sampled_from_top": 5000, "skipped_not_resolving": skipped}
    return items, meta


def freeze(n_phish: int = 240, n_tranco: int = 160, out: Optional[Path] = None) -> Path:
    from phishguard.app.capture import capture_url
    from phishguard.app.runtime_config import PipelineConfig
    from phishguard.evaluation.evaluate import provenance

    items, meta = _items(n_phish, n_tranco)
    cfg = PipelineConfig.from_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = out or (project_root() / SNAPSHOT_DIR / f"live_snapshot_{stamp}.jsonl.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {"kind": "header", "created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance(),
              "login_interaction": cfg.enable_login_interaction, "sources": meta, "n_items": len(items)}
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(header, default=str) + "\n")
        for i, it in enumerate(items):
            t0 = time.perf_counter()
            try:
                cap = capture_url(it["url"], cfg, namespace="suspicious")
                cj, err = cap.as_json(), cap.error
            except Exception as e:  # noqa: BLE001
                cj, err = None, f"{type(e).__name__}: {e}"
            html = ""
            if cj and cj.get("html_path") and Path(cj["html_path"]).is_file():
                html = Path(cj["html_path"]).read_text(encoding="utf-8", errors="ignore")[:MAX_HTML_BYTES]
            ok = cj is not None and not err and (cj or {}).get("capture_strategy") != "failed"
            fh.write(json.dumps({"kind": "item", "id": i, **it, "capture_ok": ok, "capture_error": err,
                                 "capture_seconds": time.perf_counter() - t0, "capture": cj, "html": html},
                                default=str) + "\n")
            print(f"[{i + 1}/{len(items)}] {it['set']} ok={ok} {it['url'][:80]}", flush=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")
    return out


def load_snapshot(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    header: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "header":
                header = rec
            else:
                items.append(rec)
    return header, items


def latest_snapshot() -> Path:
    files = sorted((project_root() / SNAPSHOT_DIR).glob("live_snapshot_*.jsonl.gz"))
    if not files:
        raise SystemExit(f"no snapshot in {SNAPSHOT_DIR}; run `phishguard snapshot-live` (GitHub Actions) first")
    return files[-1]


def _capture_obj(cj: Dict[str, Any], html_file: Optional[Path]):
    from phishguard.app.schemas import CaptureInteractionMetadata, CaptureResult

    d = dict(cj)
    d["interaction"] = CaptureInteractionMetadata(**(d.get("interaction") or {}))
    d["html_path"] = str(html_file) if html_file else ""
    d["screenshot_path"] = ""
    d["fullpage_screenshot_path"] = ""
    return CaptureResult(**d)


def _passes(expected: str, verdict: str) -> bool:
    from phishguard.evaluation.live import _passes as p

    return p(expected, verdict)


def replay(split: str = "val", path: Optional[Path] = None, out: Optional[Path] = None) -> Dict[str, Any]:
    import phishguard.app.dashboard as dash
    from phishguard.evaluation.evaluate import provenance

    path = path or latest_snapshot()
    header, items = load_snapshot(path)
    items = [it for it in items if split == "all" or it["split"] == split]
    os.environ["PHISH_LEGITIMACY_RESCUE_ENABLED"] = "true"
    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for it in items:
            if it.get("capture") is None:
                rows.append({k: it.get(k) for k in ("id", "set", "url", "expected", "split", "capture_ok")})
                continue
            hf = None
            if it.get("html"):
                hf = Path(td) / f"{it['id']}.html"
                hf.write_text(it["html"], encoding="utf-8")
            cap = _capture_obj(it["capture"], hf)
            with patch.object(dash, "capture_url", return_value=cap):
                payload, _ = dash.build_dashboard_analysis(it["url"], reinforcement=True)
            v = payload["verdict"]
            rows.append({"id": it["id"], "set": it["set"], "url": it["url"], "expected": it["expected"],
                         "split": it["split"], "capture_ok": it["capture_ok"],
                         "popular_domain": it.get("popular_domain"), "verdict": v["verdict_3way"],
                         "hard_blockers": v.get("evidence_hard_blockers"),
                         "layer1_p": payload["layer1_ml"].get("phish_proba"),
                         "passes": _passes(it["expected"], v["verdict_3way"]),
                         "evidence": {k: v.get(k) for k in (
                             "evidence_phishing_score", "evidence_legitimacy_score", "evidence_phishing_signals",
                             "evidence_legitimacy_signals", "evidence_ambiguity_signals",
                             "no_phishing_evidence_guard")}})
    summary = summarize(rows)
    report = {"provenance": provenance(), "command": f"phishguard replay --split {split}",
              "snapshot": {"file": str(path.relative_to(project_root())) if path.is_relative_to(project_root())
                           else str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "created_utc": header.get("created_utc")},
              "split": split, "summary": summary, "rows": rows}
    out = out or (project_root() / "metrics" / "results" / f"replay_{split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if split == "test":
        log = project_root() / "metrics" / "results" / "test_access_log.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                 "commit": report["provenance"].get("git_commit"),
                                 "dirty": report["provenance"].get("git_dirty_tracked_files"),
                                 "snapshot_sha256": report["snapshot"]["sha256"]}) + "\n")
    print(json.dumps(summary, indent=2))
    return report


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The two numbers from TUNING_PLAN.md section 4, plus per-set detail."""
    scored = [r for r in rows if "verdict" in r]
    legit = [r for r in scored if r["set"] in LEGIT_SETS]
    # Section 2 label hygiene: fresh phishing needs a successful capture and a non-popular domain.
    phish = [r for r in scored if r["set"] == "fresh_phishing" and r["capture_ok"] and not r.get("popular_domain")]
    phish_pop = [r for r in scored if r["set"] == "fresh_phishing" and r["capture_ok"] and r.get("popular_domain")]

    def count(rs, v):
        return sum(r["verdict"] == v for r in rs)

    per_set = {}
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in scored if r["set"] == s]
        per_set[s] = {"n": sum(r["set"] == s for r in rows), "scored": len(rs),
                      "capture_ok": sum(bool(r["capture_ok"]) for r in rs),
                      "verdicts": {v: count(rs, v) for v in ("likely_phishing", "uncertain", "likely_legitimate")},
                      "passes": sum(bool(r["passes"]) for r in rs)}
    return {
        "legit_false_alarms": {"n": len(legit), "likely_phishing": count(legit, "likely_phishing"),
                               "rate": count(legit, "likely_phishing") / len(legit) if legit else None},
        "phishing_recall": {"n": len(phish), "likely_phishing": count(phish, "likely_phishing"),
                            "likely_legitimate": count(phish, "likely_legitimate"),
                            "rate": count(phish, "likely_phishing") / len(phish) if phish else None},
        "phishing_on_popular_domains_reported_separately": {"n": len(phish_pop),
                                                            "likely_phishing": count(phish_pop, "likely_phishing")},
        "per_set": per_set,
    }


def main_freeze() -> None:
    ap = argparse.ArgumentParser(description="Capture every evaluation URL once and save it (needs internet).")
    ap.add_argument("--phishing", type=int, default=240)
    ap.add_argument("--tranco", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    freeze(a.phishing, a.tranco, a.out)


def main_replay() -> None:
    ap = argparse.ArgumentParser(description="Replay a frozen snapshot through the full analysis (offline).")
    ap.add_argument("--split", choices=["val", "test", "all"], default="val")
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    replay(a.split, a.snapshot, a.out)
