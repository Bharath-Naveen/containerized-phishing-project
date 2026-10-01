"""Render metrics/VERIFIED_METRICS.md and docs/MODEL_CARD.md from metrics/results/evaluation.json.

Every number in both files is read from the evaluation report; nothing is typed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

NICE = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
        "lightgbm": "LightGBM", "majority_class_baseline": "Majority-class baseline"}


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci(c, k):
    return "" if not c or k not in c else f"{c[k][0]:.3f} to {c[k][1]:.3f}"


def _table(head: List[str], rows: List[List[str]]) -> List[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _headline(r: Dict[str, Any]) -> Dict[str, Any]:
    prim = r["primary"]["model"]
    h = r["held_out_test"]["models"][prim]
    return {"primary": prim, "h": h, "base": r["held_out_test"]["models"]["majority_class_baseline"],
            "off": r["external_sets"]["official_brand_test_half"], "ph": r["external_sets"]["phishstats"],
            "dash": r["dashboard_ml_only"]["sets"]}


def _load_live():
    import json

    from phishguard.paths import project_root

    p = project_root() / "metrics" / "results" / "live_capture.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def live_section(lv: Dict[str, Any]) -> List[str]:
    p = lv["provenance"]
    L = ["", "## Full system with live page capture", "",
         f"From `phishguard evaluate-live` (GitHub Actions, commit `{p['git_commit'][:10]}`, {p['generated_utc']}). "
         f"Each page was captured once with Playwright and analyzed twice (legitimacy rescue on and off). "
         f"Dummy login interaction: {'on' if lv.get('login_interaction') else 'off'}. Raw values: `metrics/results/live_capture.json`.", ""]
    rows = []
    for n, x in lv["per_set"].items():
        v = x["verdicts"]
        rows.append([n, x["n"], f"{x['capture_ok']} ({pct(1 - x['capture_failure_rate']) if x['capture_failure_rate'] is not None else 'n/a'})",
                     v["likely_phishing"], v["uncertain"], v["likely_legitimate"], pct(x["pass_rate"]), pct(x["pass_rate_capture_ok_only"]),
                     f"{x['likely_phishing_rescue_on']} / {x['likely_phishing_rescue_off']} ({x['rescue_changed_verdicts']} changed)"])
    L += _table(["Set", "URLs", "Captured OK", "likely_phishing", "uncertain", "likely_legitimate", "Pass rate", "Pass rate (captured OK)",
                 "likely_phishing with rescue on / off"], rows)
    src = lv.get("sources") or {}
    feed = (src.get("phishing_feed") or {}).get("used")
    tr = src.get("tranco") or {}
    if src:
        L += ["", f"Fresh phishing URLs came from: {feed or 'no feed reachable'}. Popular-homepage sample: Tranco list {tr.get('list_id')}, {tr.get('skipped_not_resolving', 0)} infrastructure domains without a website skipped."]
    lat = lv["latency_seconds_full_path"]
    L += ["", "Pass means: legitimate sets not labeled likely_phishing; phishing sets labeled likely_phishing; the edge cases follow their own expected outcome (`data/evaluation/eal_edge_cases.json`). "
          "Live phishing pages are often already taken down, so check the captured-OK column before reading a phishing pass rate.", "",
          f"Full-path latency (capture + analysis): p50 {lat['p50']:.1f} s, p95 {lat['p95']:.1f} s over {lat['n']} URLs ({lat['note']}).", ""]
    if lv.get("edge_case_failures"):
        L += ["Edge cases that did not pass:", ""]
        L += _table(["URL", "Expected", "Verdict", "Captured OK"], [[f"`{x['url']}`", x["expected"], x.get("verdict"), x.get("capture_ok")] for x in lv["edge_case_failures"]])
        L += [""]
    return L


def _load_tuning():
    import json

    from phishguard.paths import project_root

    d = project_root() / "metrics" / "results" / "tuning"
    names = ("val_before", "val_after", "test_before", "test_after")
    if not all((d / f"{n}.json").is_file() for n in names):
        return None
    return {n: json.loads((d / f"{n}.json").read_text(encoding="utf-8")) for n in names}


def tuning_section(t: Dict[str, Any]) -> List[str]:
    snap = t["test_after"]["snapshot"]
    L = ["", "## Rule tuning on a frozen live snapshot", "",
         f"Pre-registered in `docs/rebuild/TUNING_PLAN.md`. Every number below replays the same saved captures "
         f"(`{snap['file']}`, sha256 `{snap['sha256'][:12]}`, captured {snap.get('created_utc', '')[:10]}) through the full analysis "
         "with no network access. Rules were changed using the val split only. The test split was scored once with the old rules and once "
         "with the final rules; one more test run, made with a stale model by mistake, is logged and discarded "
         "(`metrics/results/test_access_log.jsonl`). Model: the shipped Layer 1 bundle.", ""]
    rows = []
    for split in ("val", "test"):
        for when in ("before", "after"):
            sm = t[f"{split}_{when}"]["summary"]
            fa, rc = sm["legit_false_alarms"], sm["phishing_recall"]
            ob = sm["per_set"].get("official_brand", {})
            rows.append([split, when, f"{fa['likely_phishing']} / {fa['n']} ({pct(fa['rate'])})",
                         f"{ob.get('verdicts', {}).get('likely_phishing')} / {ob.get('n')}",
                         f"{rc['likely_phishing']} / {rc['n']} ({pct(rc['rate'])})", f"{rc['likely_legitimate']} / {rc['n']}"])
    L += _table(["Split", "Rules", "Legitimate pages labeled likely_phishing", "of which official brand",
                 "Fresh phishing labeled likely_phishing", "Fresh phishing labeled likely_legitimate"], rows)
    L += ["", "Legitimate pages: official brand URLs, popular homepages (Tranco), hard legitimate and curated legitimate URLs (all captures, failed ones included). "
          "Fresh phishing: newest PhishStats URLs whose capture succeeded and whose host is not itself a top-10K site. "
          "Feed labels mean reported as phishing; some pages were already replaced by harmless content when captured.", ""]
    return L


def render_verified(r: Dict[str, Any]) -> str:
    p = r["provenance"]
    env = p["environment"]
    d = r["data"]
    hd = _headline(r)
    prim, h = hd["primary"], hd["h"]
    L: List[str] = ["# Verified metrics", ""]
    L += ["Every number below was produced by running `phishguard evaluate` on the trained run. "
          "Nothing is copied from older reports. No synthetic data is used. Raw values: `metrics/results/evaluation.json`.", ""]
    L += [f"- Generated: {p['generated_utc']}",
          f"- Git commit: `{p['git_commit']}`" + (" (with uncommitted changes)" if p.get("git_dirty_tracked_files") else ""),
          f"- Code trees: " + ", ".join(f"{k} `{v[:10]}`" for k, v in p["code_trees"].items()),
          f"- Seed: {p['seed']} (sampling, splits, models, bootstrap)",
          f"- Environment: Python {env['python']}, {env['os']}, {env['cpu']} ({env['cpu_count']} vCPU), {env['gpu']}",
          "- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
          "- Reproduce: `phishguard train --full` then `phishguard evaluate --with-tests` (see `metrics/reproduce.sh`)", ""]

    L += ["## Headline", ""]
    L += _table(["What", "Value", "Context"], [
        [f"Primary model ({NICE[prim]}) held-out F1", f3(h["f1"]), f"95% CI {ci(h.get('bootstrap_95ci'), 'f1')}; {h['n']:,} test URLs from domains never seen in training; majority baseline F1 {f3(hd['base']['f1'])}"],
        ["ROC-AUC / PR-AUC", f"{f3(h['roc_auc'])} / {f3(h['pr_auc'])}", f"ROC-AUC 95% CI {ci(h.get('bootstrap_95ci'), 'roc_auc')}; baseline 0.500 / {f3(hd['base']['pr_auc'])}"],
        ["Precision / recall / false-positive rate", f"{f3(h['precision'])} / {f3(h['recall'])} / {pct(h['false_positive_rate'])}", "threshold 0.5 on raw model probability"],
        ["Official brand URLs flagged by Layer 1 alone", pct(hd["off"]["primary_flag_rate_app_threshold"]), f"{hd['off']['n']} real official brand URLs (test half), app threshold"],
        ["Official brand URLs the dashboard calls likely_phishing", pct(hd["dash"]["official_brand_test_half"]["false_alarm_likely_phishing_rate"]), "ML-only dashboard path (no live capture); the rest are routed to uncertain"],
        ["Real PhishStats phishing URLs flagged by Layer 1", pct(hd["ph"]["primary_flag_rate_app_threshold"]), f"{hd['ph']['n']} URLs collected 2025-04 to 2026-04, hosts excluded from training"],
        ["PhishStats URLs the dashboard calls likely_phishing", pct(hd["dash"]["phishstats"]["detected_likely_phishing_rate"]), "ML-only dashboard path"],
    ] + ([["Tests", f"{r['tests']['passed']} / {r['tests']['collected']} passed", f"{r['tests'].get('line_coverage_percent', 'n/a')}% line coverage of src/phishguard"]] if r.get("tests") else []))
    L += [""]

    L += ["## Data", ""]
    raw = d.get("raw", {})
    sp = d.get("split", {})
    ex = d.get("evaluation_exclusions", {})
    L += _table(["What", "Value"], [
        ["Source", f"{raw.get('source', 'n/a')} (`{raw.get('file', '')}`, sha256 `{str(raw.get('sha256'))[:16]}...`)"],
        ["Raw rows x columns", f"{raw.get('rows', 0):,} x {raw.get('columns', 'n/a')} ({raw.get('legit_status_1', 0):,} legitimate, {raw.get('phishing_status_0', 0):,} phishing); no date column"],
        ["After canonical dedupe", f"{d.get('deduplicated_rows') or 0:,}"],
        ["Run mode", f"{d.get('run_mode')}; {d.get('curated_legit_added') or 0} curated legitimate homepages added" + (f"; {d['tranco_rows_added']:,} Tranco homepages added" if d.get("tranco_rows_added") else "")],
        ["Removed so evaluation stays out-of-sample", f"{ex.get('rows_dropped_legit_eval_domains', 0):,} rows on legit-evaluation domains, {ex.get('rows_dropped_phish_eval_hosts', 0):,} on phishing-evaluation hosts"],
        ["Train / test rows", f"{d['train_rows']:,} / {d['test_rows']:,} (train {d['train_class_counts']['phishing']:,} phishing, test {d['test_class_counts']['phishing']:,} phishing)"],
        ["Split", f"StratifiedGroupKFold by registered domain: {sp.get('train_groups', 0):,} / {sp.get('test_groups', 0):,} domains, overlap {sp.get('registered_domain_overlap_count')}, rows without a parsable domain {sp.get('malformed_group_keys')}"],
        ["Fit / validation rows (inside train)", f"{d.get('n_fit_rows') or 0:,} / {d.get('n_validation_rows') or 0:,} (validation also grouped by domain)"],
        ["Train/serve feature parity", f"{(d.get('feature_parity_check') or {}).get('checked_rows')} rows re-extracted the app's way, {(d.get('feature_parity_check') or {}).get('mismatched_rows')} mismatches"],
        ["Model features", str(d.get("n_model_features"))],
    ])
    L += [""]

    L += ["## Held-out test (threshold 0.5)", ""]
    rows = []
    for m, x in r["held_out_test"]["models"].items():
        rows.append([NICE.get(m, m) + (" (primary)" if m == prim else ""), f3(x["precision"]), f3(x["recall"]), f3(x["f1"]),
                     ci(x.get("bootstrap_95ci"), "f1") or "n/a", f3(x["roc_auc"]), f3(x["pr_auc"]), pct(x["false_positive_rate"]),
                     f"{x['confusion_matrix']['tn']:,} / {x['confusion_matrix']['fp']:,} / {x['confusion_matrix']['fn']:,} / {x['confusion_matrix']['tp']:,}"])
    L += _table(["Model", "Precision", "Recall", "F1", "F1 95% CI", "ROC-AUC", "PR-AUC", "FPR", "TN / FP / FN / TP"], rows)
    pa = r["primary"]["app_threshold_calibrated_ge_0_5"]
    L += ["", f"Primary at the app's threshold (calibrated probability >= 0.5): precision {f3(pa['precision'])}, recall {f3(pa['recall'])}, "
          f"F1 {f3(pa['f1'])}, FPR {pct(pa['false_positive_rate'])}. Brier score raw {r['primary']['calibration']['brier_raw']:.4f}, "
          f"calibrated {r['primary']['calibration']['brier_calibrated']:.4f}.", "",
          f"Selection: {r['primary']['selection_rule']}. Validation PR-AUC per model: "
          + ", ".join(f"{NICE[m]} {f3((x.get('validation') or {}).get('val_pr_auc'))}" for m, x in r["held_out_test"]["models"].items() if m in NICE and m != "majority_class_baseline") + ".", ""]

    cv = r["cross_validation"]
    L += [f"## Cross-validation ({cv['method']}, {cv['rows']:,} rows)", ""]
    L += _table(["Model", "F1", "ROC-AUC", "PR-AUC", "FPR"], [
        [NICE.get(m, m), f"{x['f1']['mean']:.3f} ± {x['f1']['std']:.3f}", f"{x['roc_auc']['mean']:.3f} ± {x['roc_auc']['std']:.3f}",
         f"{x['pr_auc']['mean']:.3f} ± {x['pr_auc']['std']:.3f}", f"{100 * x['false_positive_rate']['mean']:.1f}% ± {100 * x['false_positive_rate']['std']:.1f}"]
        for m, x in cv["models"].items()])
    L += ["", f"Mean ± standard deviation over 5 folds; domain overlap per fold: {cv['group_overlap_per_fold']}. The majority-class baseline predicts whichever class is larger in each training fold, so in a fold where phishing is the majority it flags everything.", ""]

    L += ["## External sets (never in training)", ""]
    rows = []
    for n, x in r["external_sets"].items():
        rows.append([n, x["n"], x["label"], pct(x["primary_flag_rate_app_threshold"]),
                     ", ".join(f"{NICE[m]} {pct(v)}" for m, v in x["layer1_flag_rate_by_model_raw_0_5"].items()),
                     f"{x['urls_whose_domain_is_in_training']} domain / {x['urls_whose_host_is_in_training']} host"])
    L += _table(["Set", "URLs", "Truth", "Primary flags (app threshold)", "Each model flags (raw 0.5)", "Overlap with training"], rows)
    L += ["", "For legitimate sets the flag rate is the false-alarm rate; for phishing sets it is recall. The overlap column counts URLs whose registered domain or exact host also appears in the training rows. The official-brand validation half was used to break model-selection ties, so quote the test half.", ""]

    L += [f"## Dashboard verdicts ({r['dashboard_ml_only']['mode']})", ""]
    rows = []
    for n, x in r["dashboard_ml_only"]["sets"].items():
        v = x["verdicts"]
        key = "detected_likely_phishing_rate" if "detected_likely_phishing_rate" in x else "false_alarm_likely_phishing_rate"
        rows.append([n, x["n"], v["likely_phishing"], v["uncertain"], v["likely_legitimate"],
                     ("detected " if key.startswith("detected") else "false alarms ") + pct(x[key]), pct(x["layer1_flag_rate"])])
    L += _table(["Set", "URLs", "likely_phishing", "uncertain", "likely_legitimate", "Rate", "Layer 1 flag rate"], rows)
    L += ["", "Without live capture the adjudication layer treats missing page evidence as a reason for `uncertain`, not as proof of safety.", ""]

    lat = r["latency_ms"]
    L += ["## Latency", ""]
    L += _table(["Path", "p50", "p95", "Context"], [
        ["Layer 1 only (features + model + calibration)", f"{lat['layer1_only']['p50']:.1f} ms", f"{lat['layer1_only']['p95']:.1f} ms", f"{lat['n_urls']} held-out URLs; {lat['hardware']}; {lat['note']}"],
        ["Dashboard, ML-only (all rules + EAL)", f"{lat['dashboard_ml_only_incl_eal']['p50']:.1f} ms", f"{lat['dashboard_ml_only_incl_eal']['p95']:.1f} ms", "same URLs"],
    ])
    live = _load_live()
    if live:
        L += live_section(live)
    else:
        L += ["", "## Not run here", ""]
        L += _table(["Item", "Why"], [[x["item"], x["reason"]] for x in r["not_run"]])
    tuning = _load_tuning()
    if tuning:
        L += tuning_section(tuning)
    L += ["", "## Earlier baseline", "",
          "The pre-rebuild audit (leaky split, scheme artifact, stale calibrator) is kept for comparison in `metrics/audit_baseline/` and at git tag `audit-baseline-2026-09-29`. Its numbers describe the old code and must not be quoted for the current system.", ""]
    return "\n".join(L)


def render_model_card(r: Dict[str, Any]) -> str:
    hd = _headline(r)
    prim, h = hd["primary"], hd["h"]
    d = r["data"]
    p = r["provenance"]
    L = [f"# Model card: phishguard Layer-1 URL model ({NICE[prim]})", "",
         f"Generated by `phishguard evaluate` on {p['generated_utc']} at commit `{p['git_commit'][:10]}`. All numbers come from `metrics/results/evaluation.json`.", "",
         "## What it is", "",
         f"{'An' if NICE[prim][0] in 'AEIOUX' else 'A'} {NICE[prim]} classifier that scores a URL's text (no page visit) for phishing. It is Layer 1 of a larger system: its score is one input to a deterministic Evidence Adjudication Layer that also weighs live-page, HTML and brand evidence before giving a verdict. It should not be used on its own to block sites.", "",
         "## Intended use", "",
         "- Fast first-pass triage of URLs inside the phishguard dashboard.",
         "- Research and demonstration of explainable phishing detection.",
         "", "Not intended for: blocking or allow-listing without the adjudication layer, or for URLs whose meaning depends on page content (for example a compromised legitimate site).", "",
         "## Training data", "",
         f"- Kaggle `harisudhan411/phishing-and-legitimate-urls`: {d.get('raw', {}).get('rows', 0):,} URLs, deduplicated to {d.get('deduplicated_rows') or 0:,}, plus {d.get('curated_legit_added') or 0} curated legitimate homepages. No timestamps, so drift over time cannot be measured from this data.",
         f"- Split by registered domain: {d['train_rows']:,} training and {d['test_rows']:,} test URLs, no domain in both.",
         "- Evaluation URLs (curated suites, official brand URLs, PhishStats URLs) are removed from training.",
         f"- {d.get('n_model_features')} URL and host features, computed scheme-neutral (the model never sees http vs https, which is an artifact of this dataset).", "",
         "## Performance", "",
         f"- Held-out test: F1 {f3(h['f1'])} (95% CI {ci(h.get('bootstrap_95ci'), 'f1')}), ROC-AUC {f3(h['roc_auc'])}, PR-AUC {f3(h['pr_auc'])}, precision {f3(h['precision'])}, recall {f3(h['recall'])}, false-positive rate {pct(h['false_positive_rate'])}. Majority-class baseline: F1 {f3(hd['base']['f1'])}, ROC-AUC 0.500.",
         f"- Real official brand URLs (not in training): Layer 1 flags {pct(hd['off']['primary_flag_rate_app_threshold'])} of {hd['off']['n']}; the dashboard labels {pct(hd['dash']['official_brand_test_half']['false_alarm_likely_phishing_rate'])} as likely_phishing (the rest go to uncertain).",
         f"- Real PhishStats phishing URLs (hosts not in training): Layer 1 flags {pct(hd['ph']['primary_flag_rate_app_threshold'])} of {hd['ph']['n']}.", "",
         "## Limitations", "",
         "- The legitimate side of the training data does not look like many modern official sites, so the URL model alone over-flags them. The adjudication layer is what keeps those from becoming phishing verdicts.",
         "- URL-only features cannot see compromised legitimate domains or page content.",
         "- The Kaggle data has no dates; performance on future phishing campaigns is unmeasured beyond the PhishStats sample.",
         "- Without live capture the system rarely returns likely_legitimate; that is by design.", "",
         "## Reproduce", "", "```", "phishguard train --full", "phishguard evaluate --with-tests", "```", ""]
    return "\n".join(L)


def write_markdown(report: Dict[str, Any], verified_path: Path, card_path: Path) -> None:
    verified_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    verified_path.write_text(render_verified(report), encoding="utf-8")
    card_path.write_text(render_model_card(report), encoding="utf-8")


def main() -> None:
    """Re-render the markdown from an existing evaluation.json (no re-measuring)."""
    import argparse
    import json

    from phishguard.paths import project_root

    ap = argparse.ArgumentParser(description="Render VERIFIED_METRICS.md and MODEL_CARD.md from evaluation.json")
    ap.add_argument("evaluation_json", nargs="?", default=str(project_root() / "metrics" / "results" / "evaluation.json"))
    a = ap.parse_args()
    report = json.loads(Path(a.evaluation_json).read_text(encoding="utf-8"))
    write_markdown(report, project_root() / "metrics" / "VERIFIED_METRICS.md", project_root() / "docs" / "MODEL_CARD.md")


if __name__ == "__main__":
    main()
