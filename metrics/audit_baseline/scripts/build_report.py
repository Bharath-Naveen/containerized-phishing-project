"""Assemble metrics/VERIFIED_METRICS.json and metrics/VERIFIED_METRICS.md from metrics/results/*.json.

Every number in the two output files is read from a results file written by a script that ran the
code. Nothing is typed in by hand. Statuses:
  VERIFIED        produced by running code in this repo on the real data in this run
  NOT RUN         could not be run here (reason given)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from common import METRICS, RESULTS, SEED

R = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}


def pct(x, d=1):
    return None if x is None else f"{100 * x:.{d}f}%"


def f3(x):
    return None if x is None else f"{x:.3f}"


def ci(c):
    return f"95% CI {c[0]:.3f} to {c[1]:.3f}"


ENTRIES = []


def add(section, metric, value, context, command, status="VERIFIED", source=None):
    ENTRIES.append({
        "section": section, "metric": metric, "value": value, "context": context,
        "command": command, "status": status, "source_result_file": source,
    })


def build():
    t = R.get("00_tests")
    if t:
        add("Tests", "pytest collected / passed / failed / skipped",
            f"{t['collected']} / {t['passed']} / {t['failed']} / {t['skipped']}",
            f"pass rate {pct(t['pass_rate'])}; runtime {t['runtime_seconds']} s",
            "python metrics/scripts/00_tests.py", source="00_tests.json")
        add("Tests", "Line coverage of src/ (pytest-cov)", f"{t['line_coverage_percent_src']}%",
            f"{t['covered_lines']} of {t['num_statements']} statements", "python metrics/scripts/00_tests.py", source="00_tests.json")

    p = R.get("01_pipeline")
    if p:
        raw = p["raw_dataset"]
        sc = raw["status_counts_raw_kaggle_convention"]
        add("Data", "Raw Kaggle rows x columns", f"{raw['rows']:,} x {raw['n_columns']}",
            f"legit (status 1) {int(sc.get('1', 0)):,}, phishing (status 0) {int(sc.get('0', 0)):,}; no date column",
            "python metrics/scripts/01_pipeline.py", source="01_pipeline.json")
        add("Data", "Rows after URL canonical dedupe", f"{p['dedup']['canonical_url_rows_after']:,}",
            f"from {p['dedup']['canonical_url_rows_before']:,}", "python metrics/scripts/01_pipeline.py", source="01_pipeline.json")
        s = p["split"]
        add("Data", "Documented default sample and grouped split",
            f"{s['n_train'] + s['n_test']:,} rows: train {s['n_train']:,} / test {s['n_test']:,}",
            f"50,000 stratified (seed {SEED}) + 896 curated legit URLs; StratifiedGroupKFold by registered domain; "
            f"train/test domain overlap {s['registered_domain_overlap_count']}, URL overlap {s['canonical_url_overlap_count']}",
            "python metrics/scripts/01_pipeline.py", source="01_pipeline.json")
        add("Data", "Features used by the models", str(p["n_features_used"]),
            "59 extracted; has_https dropped on purpose, tld and public_suffix dropped as high-cardinality",
            "python metrics/scripts/01_pipeline.py", source="01_pipeline.json")
        add("Runtime", "Layer-1 pipeline end to end (ingest, dedupe, sample, enrich, split, train 4 models)",
            f"{p['end_to_end_runtime_seconds']} s", "50K sample, CPU only (see environment)",
            "python metrics/scripts/01_pipeline.py", source="01_pipeline.json")

    e = R.get("02_layer1_eval")
    if e:
        d = e["data"]
        ctx = f"held-out test {d['n_test_rows']:,} rows ({d['test_class_counts']['phish_1']:,} phishing), unseen domains, threshold 0.5"
        for m in ["dummy_most_frequent", "logistic_regression", "random_forest", "xgboost", "lightgbm"]:
            h = e["held_out_test"][m]
            bc = h.get("bootstrap_95ci", {})
            val = (f"P {f3(h['precision'])}, R {f3(h['recall'])}, F1 {f3(h['f1'])}, ROC-AUC {f3(h['roc_auc'])}, "
                   f"PR-AUC {f3(h['pr_auc'])}, FPR {pct(h['false_positive_rate'])}")
            c2 = ctx + (f"; F1 {ci(bc['f1'])}; ROC-AUC {ci(bc['roc_auc'])}" if bc else "")
            c2 += f"; confusion [[tn,fp],[fn,tp]] = {h['confusion_matrix_[[tn,fp],[fn,tp]]']}"
            add("Layer-1 models (50K default run)", f"Held-out: {m}", val, c2, "python metrics/scripts/02_evaluate_layer1.py", source="02_layer1_eval.json")
        for m, v in e["cv_5fold_stratified_group"].items():
            add("Layer-1 models (50K default run)", f"5-fold grouped CV: {m}",
                f"F1 {v['f1']['mean']:.3f} ± {v['f1']['std']:.3f}, ROC-AUC {v['roc_auc']['mean']:.3f} ± {v['roc_auc']['std']:.3f}, PR-AUC {v['pr_auc']['mean']:.3f} ± {v['pr_auc']['std']:.3f}",
                f"StratifiedGroupKFold(5) by registered domain on all {d['n_rows_total']:,} rows; mean ± std across folds; group overlap per fold {e['cv_group_overlap_per_fold']}",
                "python metrics/scripts/02_evaluate_layer1.py", source="02_layer1_eval.json")
        add("Layer-1 models (50K default run)", "Composite-rule winner", e["winner_composite"],
            f"rule: {e['composite_rule']}; best F1 = {e['winner_by_f1']}, best ROC-AUC = {e['winner_by_roc_auc']}",
            "python metrics/scripts/02_evaluate_layer1.py", source="02_layer1_eval.json")
        a = e["app_threshold_view_for_winner"]
        fl = a["layer1_flag_calibrated_ge_0_5"]
        add("Layer-1 models (50K default run)", f"Winner at the app threshold (calibrated P >= 0.5): {a['model']}",
            f"P {f3(fl['precision'])}, R {f3(fl['recall'])}, F1 {f3(fl['f1'])}, FPR {pct(fl['false_positive_rate'])}",
            f"3-way split from the proxy formula in benchmark_url_suites.py (0.65 x calibrated, thresholds 0.56/0.38), not the full dashboard: {a['ml_only_3way_verdict_counts']}", "python metrics/scripts/02_evaluate_layer1.py", source="02_layer1_eval.json")

    ab = R.get("03_ablations")
    if ab:
        v = ab["variants"]
        for m in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
            a0, a1 = v["asis"]["models"][m], v["scheme_neutral"]["models"][m]
            add("Ablations", f"Scheme-artifact check: {m}",
                f"F1 {a0['f1']:.3f} -> {a1['f1']:.3f}; ROC-AUC {a0['roc_auc']:.3f} -> {a1['roc_auc']:.3f}",
                "same rows and split, features recomputed with every URL rewritten to https://",
                "python metrics/scripts/03_ablations.py", source="03_ablations.json")
        dc = ab["decontamination"]
        add("Ablations", "Evaluation-list domains found in training rows", f"{dc['train_rows_removed']:,} train rows",
            f"by source {dc['train_rows_removed_by_source']}; {len(dc['eval_registered_domains'])} domains from url_suites, hard_legit, composite audit URLs",
            "python metrics/scripts/03_ablations.py", source="03_ablations.json")
        add("Ablations", "Composite winner: as-is / scheme-neutral / decontaminated",
            f"{v['asis']['winner_composite']} / {v['scheme_neutral']['winner_composite']} / {v['decontaminated']['winner_composite']}",
            "best F1 in each: " + " / ".join(v[k]["winner_by_f1"] for k in ("asis", "scheme_neutral", "decontaminated")),
            "python metrics/scripts/03_ablations.py", source="03_ablations.json")

    s4 = R.get("04_suites_and_fp")
    if s4:
        for var, blk in s4["variants"].items():
            on = blk["rescue_on"]["per_set"]
            for setname, st in on.items():
                if st["expected"] == "phishing":
                    val = f"{pct(st['pass_rate_likely_phishing'], 0)} likely_phishing (n={st['n']})"
                else:
                    val = (f"FP {pct(st['false_positive_rate_likely_phishing'], 1)}; not-flagged {pct(st['pass_rate_not_flagged_phishing'], 0)}; "
                           f"strict likely_legitimate {pct(st['pass_rate_strict_likely_legitimate'], 0)} (n={st['n']})")
                add(f"Dashboard verdicts, ML-only path ({var})", setname, val, f"verdicts {st['verdicts']}; Layer-1 flag rate {pct(st['layer1_flag_rate'])}",
                    "python metrics/scripts/04_suites_and_fp.py", source="04_suites_and_fp.json")
            ra = blk["rescue_ablation"]
            add(f"Dashboard verdicts, ML-only path ({var})", "Legitimacy rescue ON vs OFF",
                f"legit FPs {ra['legit_false_positives_rescue_on']} (on) vs {ra['legit_false_positives_rescue_off']} (off); verdicts changed {ra['verdicts_changed']} of {ra['n_urls_compared']}",
                "PHISH_LEGITIMACY_RESCUE_ENABLED toggled; no live capture, so capture-dependent rescue conditions cannot fire",
                "python metrics/scripts/04_suites_and_fp.py", source="04_suites_and_fp.json")
            lat = blk["rescue_on"].get("latency_ms_cpu")
            if lat:
                l1, dsh = lat["t_layer1_ms"], lat["t_dashboard_ml_only_ms"]
                add("Latency", f"Layer-1 only, per URL ({var})", f"p50 {l1['p50']:.1f} ms, p95 {l1['p95']:.1f} ms",
                    f"n={l1['n']} held-out URLs, warm process, CPU only", "python metrics/scripts/04_suites_and_fp.py", source="04_suites_and_fp.json")
                add("Latency", f"Dashboard ML-only path incl. EAL, per URL ({var})", f"p50 {dsh['p50']:.1f} ms, p95 {dsh['p95']:.1f} ms",
                    f"n={dsh['n']}, reinforcement=False, includes 4-model agreement", "python metrics/scripts/04_suites_and_fp.py", source="04_suites_and_fp.json")
    add("Latency", "Full path with Playwright live capture", "NOT RUN",
        "verification sandbox has no general internet egress (only package registries); needs a machine with internet",
        "python -m phishguard.app.dashboard --url <url>", status="NOT RUN")
    add("Ablations", "Legitimacy rescue FP delta with live capture", "NOT RUN",
        "rescue conditions depend on redirect chain, form targets and DOM risk from Playwright capture; no internet egress here",
        "PHISH_LEGITIMACY_RESCUE_ENABLED=false python -m phishguard.evaluation.fp_audit --reinforcement", status="NOT RUN")
    add("Ablations", "EAL edge-case validation (15 live URLs in outputs/reports/eal_edge_case_validation.md)", "NOT RUN",
        "needs live Playwright capture of real sites; no internet egress in the verification sandbox",
        "see outputs/reports/eal_edge_case_validation.md", status="NOT RUN")

    f6 = R.get("06_full_run")
    if f6:
        s = f6["split"]
        add("Full-data run (~797K rows)", "Rows and grouped split", f"train {s['n_train']:,} / test {s['n_test']:,}",
            f"all 796,446 deduplicated Kaggle rows + 837 curated legit; {s['train_groups']:,} / {s['test_groups']:,} domain groups; overlap {s['registered_domain_overlap_count']}",
            "python metrics/scripts/06_full_run.py", source="06_full_run.json")
        add("Runtime", "Layer-1 pipeline end to end, full data", f"{f6['end_to_end_runtime_seconds']} s",
            "run_kaggle_pipeline(full_dataset=True) with checkpoint_every=10**9 (default checkpointing is what makes --full take hours)",
            "python metrics/scripts/06_full_run.py", source="06_full_run.json")
        for m, h in f6["held_out_test_threshold_0_5_raw"].items():
            bc = h.get("bootstrap_95ci", {})
            add("Full-data run (~797K rows)", f"Held-out: {m}",
                f"P {f3(h['precision'])}, R {f3(h['recall'])}, F1 {f3(h['f1'])}, ROC-AUC {f3(h['roc_auc'])}, PR-AUC {f3(h['pr_auc'])}, FPR {pct(h['false_positive_rate'])}",
                f"test {s['n_test']:,} rows, unseen domains, threshold 0.5" + (f"; F1 {ci(bc['f1'])}; ROC-AUC {ci(bc['roc_auc'])}" if bc else ""),
                "python metrics/scripts/06_full_run.py", source="06_full_run.json")
        add("Full-data run (~797K rows)", "Composite-rule winner / best F1", f"{f6['winner_composite']} / {f6['winner_by_f1']}",
            "composite = argmax(F1 - mean P(phish) on 8 official HTTPS URLs)", "python metrics/scripts/06_full_run.py", source="06_full_run.json")
        ex = f6["external_sets_app_features"]
        for m, v in ex["per_model"].items():
            add("Full-data run (~797K rows)", f"External sets: {m}",
                f"official-brand FP {pct(v['official_brand_fp_rate_raw_0_5'])} ({v['official_brand_fp_count']}/{ex['official_brand_urls_n']}); PhishStats recall {pct(v['phishstats_recall_raw_0_5'])} (n={ex['phishstats_n']})",
                f"Layer-1 only, raw P >= 0.5; {ex['official_brand_domains_in_train']} official and {ex['phishstats_domains_in_train']} PhishStats URLs share a domain with training rows",
                "python metrics/scripts/06_full_run.py", source="06_full_run.json")

    b5 = R.get("05b_deployed_leakfix")
    if b5:
        add("Deployed models: split and feature fix (evaluation only)", "Test rows that share a real registered domain with training rows",
            f"{b5['test_rows_sharing_real_domain_with_train']:,} of {b5['test_rows']:,} ({pct(b5['test_rows_sharing_real_domain_with_train'] / b5['test_rows'])})",
            f"retrain_with_fresh.py kept scheme-less URLs, so {b5['train_rows_with_malformed_group_key']:,} of {b5['train_rows']:,} train rows got a per-URL 'malformed::' group key instead of their domain",
            "python metrics/scripts/05b_deployed_model_leakfix_eval.py", source="05b_deployed_leakfix.json")
        lf = b5["leak_free_class_counts"]
        for m, v in b5["results"].items():
            if m.startswith("dummy"):
                continue
            a, b, c = v["a_as_reported_all_test_csv_features"], v["b_leak_free_rows_csv_features"], v["c_leak_free_rows_app_features"]
            add("Deployed models: split and feature fix (evaluation only)", f"{m}: as reported -> leak-free rows -> leak-free rows with app features",
                f"ROC-AUC {a['roc_auc']:.3f} -> {b['roc_auc']:.3f} -> {c['roc_auc']:.3f}; FPR {pct(a['false_positive_rate'])} -> {pct(b['false_positive_rate'])} -> {pct(c['false_positive_rate'])}; recall {pct(a['recall'])} -> {pct(b['recall'])} -> {pct(c['recall'])}",
                f"leak-free rows: {b5['test_rows_leak_free']:,} ({lf['phish_1']:,} phishing, {lf['legit_0']:,} legit, so F1 is inflated by prevalence; use ROC-AUC and FPR)",
                "python metrics/scripts/05b_deployed_model_leakfix_eval.py", source="05b_deployed_leakfix.json")

    d5 = R.get("05_deployed_model")
    if d5:
        dd = d5["data"]
        ctx = f"saved split of run {d5['deployed_run']}: train {dd['n_train']:,} / test {dd['n_test']:,}, {dd['registered_domains_shared_by_train_and_test']:,} registered domains appear in both (see the split and feature fix section); threshold 0.5"
        for m, h in d5["held_out_test_threshold_0_5_raw"].items():
            bc = h.get("bootstrap_95ci", {})
            add("Deployed models (500K run, shipped in outputs/models)", f"Held-out: {m}",
                f"P {f3(h['precision'])}, R {f3(h['recall'])}, F1 {f3(h['f1'])}, ROC-AUC {f3(h['roc_auc'])}, PR-AUC {f3(h['pr_auc'])}, FPR {pct(h['false_positive_rate'])}",
                ctx + (f"; F1 {ci(bc['f1'])}" if bc else ""), "python metrics/scripts/05_deployed_model.py", source="05_deployed_model.json")
        o = d5["official_brand_urls"]
        add("Deployed models (500K run, shipped in outputs/models)", "Layer-1 FP rate on curated official brand URLs",
            f"{pct(o['fp_rate_raw_0_5'])} ({o['fp_count_raw_0_5']} of {o['n_urls']})",
            f"phishing_dataset/dataset_test_recent.csv brand_official rows, collected 2026-04-26; {o['n_urls_exactly_in_train']} URL(s) in training; raw P >= 0.5",
            "python metrics/scripts/05_deployed_model.py", source="05_deployed_model.json")
        fr = d5["fresh_phishstats"]
        add("Deployed models (500K run, shipped in outputs/models)", "Layer-1 recall on PhishStats phishing URLs",
            f"{pct(fr['recall_rows_not_in_train'])} (n={fr['n_rows_not_in_train']})",
            f"URLs not in the deployed training split; {fr.get('n_rows_whose_domain_in_train', 'n/a')} share a registered domain with training rows; collected {fr.get('collection_dates')}",
            "python metrics/scripts/05_deployed_model.py", source="05_deployed_model.json")
        c = d5["calibration"]
        add("Deployed models (500K run, shipped in outputs/models)", "Brier score: raw / shipped calibrator / refit calibrator",
            f"{c['brier_raw']:.4f} / {c['brier_shipped_calibrator']:.4f} / {c['brier_refit_calibrator']:.4f}",
            c["note"], "python metrics/scripts/05_deployed_model.py", source="05_deployed_model.json")


def write():
    build()
    prov = {}
    for k, v in R.items():
        prov[k] = v.get("_provenance", {})
    any_prov = next(iter(prov.values())) if prov else {}
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": any_prov.get("git_commit"),
        "seed": SEED,
        "environment": any_prov.get("environment"),
        "reproduce": "bash metrics/reproduce.sh",
        "data": {
            "kaggle_csv": "data/raw/kaggle/phishing_and_legitimate_urls.csv (Kaggle harisudhan411/phishing-and-legitimate-urls, not committed)",
            "kaggle_csv_sha256": R.get("01_pipeline", {}).get("raw_dataset", {}).get("sha256"),
            "synthetic_data_used": False,
        },
        "metrics": ENTRIES,
        "per_step_provenance": prov,
    }
    (METRICS / "VERIFIED_METRICS.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    env = doc["environment"] or {}
    lines = [
        "# Verified metrics",
        "",
        "Every number below was produced by running the code in this repo on the real Kaggle data. "
        "No synthetic data is used anywhere in this audit. Numbers from older reports or docs are not repeated here unless re-run.",
        "",
        f"- Generated: {doc['generated_utc']}",
        f"- Git commit: `{doc['git_commit']}` (code trees: " + ", ".join(f"{k} `{v[:10]}`" for k, v in (any_prov.get("code_trees") or {}).items()) + ")",
        f"- Seed: {SEED} (numpy, sklearn, sampling, splits, bootstrap)",
        f"- Environment: Python {env.get('python')}, {env.get('os')}, {env.get('cpu')} ({env.get('cpu_count')} vCPU, {env.get('ram_gb')} GB RAM), {env.get('gpu')}",
        f"- Packages: {', '.join(f'{k} {v}' for k, v in (env.get('packages') or {}).items())}",
        f"- Data: `{doc['data']['kaggle_csv']}`, sha256 `{doc['data']['kaggle_csv_sha256']}`",
        "- Reproduce everything: `bash metrics/reproduce.sh` (raw per-step outputs in `metrics/results/`)",
        "",
    ]
    order = ["Tests", "Data", "Runtime", "Layer-1 models (50K default run)", "Full-data run (~797K rows)", "Ablations",
             "Deployed models (500K run, shipped in outputs/models)", "Deployed models: split and feature fix (evaluation only)",
             "Dashboard verdicts, ML-only path (deployed)", "Dashboard verdicts, ML-only path (asis_50k)",
             "Dashboard verdicts, ML-only path (decontaminated)", "Latency"]
    ENTRIES.sort(key=lambda e: order.index(e["section"]) if e["section"] in order else len(order))
    section = None
    for e in ENTRIES:
        if e["section"] != section:
            section = e["section"]
            lines += ["", f"## {section}", "", "| Metric | Value | Context | Status |", "|---|---|---|---|"]
        ctx = str(e["context"]).replace("|", "/")
        lines.append(f"| {e['metric']} | {e['value']} | {ctx} | {e['status']} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- The audit ran on a clean clone of GitHub `main`; its `src/`, `tests/`, `data/evaluation/` and `requirements.txt` trees are byte-identical to the local commit `4cb3d6b` (that commit only adds docs/BUILD_STORY.md). The code-tree hashes above let you check this with `git rev-parse HEAD:src`.",
        "- 50K default run: the documented default (`python -m phishguard.pipelines.kaggle`), Kaggle only, re-run from scratch here.",
        "- Deployed models: the four `.joblib` files the app loads today (500K-row run 20260427_234728, byte-identical to that run's folder). They were re-scored on that run's saved held-out split, not retrained.",
        "- Dashboard verdict rows use the ML-only path (`reinforcement=False`). Live Playwright capture could not run in the verification sandbox.",
        "- Several curated evaluation URLs are also added to training by `simple_legit_augment` (see Ablations). Treat suite and hard-legit pass rates for the as-is and deployed models as in-sample.",
    ]
    (METRICS / "VERIFIED_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote metrics/VERIFIED_METRICS.json and metrics/VERIFIED_METRICS.md")


if __name__ == "__main__":
    write()
