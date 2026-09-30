# How the project works (current state, after rebuild Phase 2)

A plain-language map of the system. It is updated as the rebuild changes things; the dated history of each change is in [REBUILD_LOG.md](REBUILD_LOG.md).

## The one-paragraph version

You give it a URL. A fast machine-learning model scores the URL text alone (Layer 1). Optionally, a headless browser opens the page and collects evidence: redirects, forms, TLS state, network requests (Layer 2). The HTML, the host and path, and brand-to-domain fit are then analyzed (Layer 3 and brand context). Finally a rule-based judge, the Evidence Adjudication Layer (EAL), weighs everything and returns `likely_phishing`, `uncertain` or `likely_legitimate`, with reasons. The ML score alone never decides; the EAL does.

## Part 1: How a URL is scored (runtime, `src/phishguard/app/`)

Try it: `phishguard analyze --url "https://example.com" --no-reinforcement` (ML-only) or `phishguard serve` (dashboard).


| Step | What happens | Where (under `src/phishguard/`) |
|---|---|---|
| 1. Normalize | The URL is put in one canonical form (lowercase host, scheme added if missing, trailing slash rules). Features are computed from a scheme-neutral copy (`http://` forced), so `https://x.com` and `x.com` look identical to the model. | `urls/normalize.py` |
| 2. Layer 1 features | 59 cheap features from the URL text only: lengths, dots, digits, entropy, suspicious words, brand tokens in the wrong place, free-hosting domains, official-domain registry match. No network. 54 of them are used by the model (3 scheme-derived ones and 2 high-cardinality text columns are left out). | `features/layer1.py`, `features/` |
| 3. Layer 1 model | The **model bundle** (`outputs/models/layer1_bundle.joblib`) holds the trained pipeline, its own probability calibrator and the exact feature list. Output: raw and calibrated P(phishing). | `app/ml_layer1.py` |
| 4. Model agreement | Four "witness" models (LR, RF, XGBoost, LightGBM) also score the URL; their votes become a consensus signal (strong_phishing, split, and so on). Supporting evidence only. | `app/ml_layer1.py` (`compute_layer1_model_agreement`) |
| 5. Layer 2 capture (optional) | Playwright loads the page: final URL, redirect chain, form targets, TLS/cert state, network requests, HTML. Skipped in "ML-only" mode. | `app/capture.py`, `app/capture_signals.py` |
| 6. Layer 3 analysis | HTML structure and DOM anomalies (login harvesters, wrappers), JS/network behavior, host/path reasoning (is this host shape suspicious? does the path fit?). | `app/html_*_signals.py`, `app/behavior_signals.py`, `app/host_path_reasoning.py` |
| 7. Brand and trust context | Does the page's brand match the domain? Is the domain in the small official-domain registry (a weak prior, not a whitelist)? Is it user-hosted (github.io, netlify.app)? | `app/brand_coherence.py`, `app/registries.py`, `data/official_domains.json`, `data/reference/*.csv` |
| 8. EAL verdict | Adds up phishing, legitimacy and ambiguity signals, applies hard blockers (for example credential form posting to another domain), and picks the verdict. When evidence is missing or conflicting it says `uncertain` on purpose. | `app/verdict_rules.py` (pre-EAL adjustments), `app/eal.py` (the judge), `app/dashboard.py` (wires it all together) |

Important current behavior: without Layer 2 capture, the EAL almost never says `likely_legitimate` (missing evidence is not treated as proof of safety). That is by design, and it is why the demo and metrics distinguish "ML-only" from "full capture".

## Part 2: How the model is trained (`src/phishguard/data`, `features`, `models`, `pipelines`)

Entry point: `phishguard train` (default 50,000-row sample; `--full` for all rows). The code is `pipelines/kaggle.py`, which calls the steps below.

1. **Ingest** the Kaggle CSV (`url,status`; Kaggle `status 1` = legit is mapped to internal `label 0` = legit, `1` = phishing).
2. **Clean**: canonicalize every URL and drop exact duplicates (822,010 rows become about 796K).
3. **Sample**: stratified sample (keeps the class balance), seed 42 (`data/sample.py`).
4. **Augment**: add ~880 curated legitimate homepages (`data/evaluation/simple_legit_urls.jsonl`).
5. **Remove evaluation URLs** (new): any row on the registered domain of a legitimate evaluation URL, or on the exact host of a phishing evaluation URL, is dropped, so evaluation stays out-of-sample.
6. **Enrich**: compute the Layer 1 features for every row.
7. **Split**: train/test by registered domain (StratifiedGroupKFold, `data/split.py`), so a website is never in both. Guards (`guards.py`) stop the run if too many rows lack a real domain or if any domain overlaps.
8. **Train** (`models/train.py`): four models. Inside the training rows, a second domain-grouped split makes a validation set. A parity guard re-computes features for 300 stored rows the way the app does and stops the run on any mismatch.
9. **Select** the primary model with the `validated` rule: best PR-AUC on the validation set, tie broken by fewer false alarms on the validation half of the official-brand URLs. The test set is never used to choose.
10. **Calibrate** probabilities on the validation set and write the **bundle** (model + calibrator + features + provenance: train-data hash, row counts, seed, date).

## Part 3: Evaluation data (`data/evaluation/`)

| File | What | Used for |
|---|---|---|
| `url_suites.json` | 18 hand-picked URLs in 4 buckets (obvious/tricky legit, obvious/hard phishing) | quick regression checks, demo |
| `hard_legit_urls.jsonl` | 15 tricky legitimate login/dashboard URLs | false-positive checks |
| `official_brand_urls.jsonl` | 298 real official brand URLs (31 domains), split by domain into `val` (162, model selection) and `test` (136, reporting only) | false-alarm rate on real legit sites |
| `phishstats_urls.jsonl` | 284 real phishing URLs from the PhishStats feed (2025-04 to 2026-04) | recall on real, recent phishing |
| `simple_legit_urls.jsonl` | ~880 curated legit homepages | training augmentation (minus any evaluation domains) |

## Part 4: Verified numbers

`metrics/` re-runs everything with seed 42 and writes each number with its command, commit and environment. `metrics/VERIFIED_METRICS.md` is the audit baseline (before the rebuild); `metrics/results/10_phase1_check.json` is the Phase 1 progress check. Only numbers from these files go on a resume or website.

## Part 5: Code map

```text
src/phishguard/
  cli.py          the `phishguard` command (train, analyze, evaluate, deploy, serve)
  config.py       seed 42 and other defaults
  paths.py        where data/, outputs/, logs/ live (PHISH_* env vars override)
  guards.py       checks that stop a leaky split or train/serve skew
  urls/           safe.py (never-crash URL parsing), normalize.py (canonical + scheme-neutral)
  data/           kaggle.py, clean.py, sample.py, augment.py, eval_sets.py, enrich.py, split.py, labels.py, fresh_*.py
  features/       layer1.py (the 59 features) built from url_features, hosting_features, brand_signals
  models/         train.py (4 models, validated selection, calibrator, bundle), deploy.py
  pipelines/      kaggle.py (main training pipeline), retrain_with_fresh.py (Kaggle + PhishStats/Tranco)
  evaluation/     url_suites.py, fp_audit.py, phish_audit.py, legit_audit.py, leakage_report.py, dataset_report.py
  app/
    dashboard.py        build_dashboard_analysis(): runs every layer in order, returns the JSON
    ml_layer1.py        loads the model bundle, scores the URL, 4-model agreement
    capture.py          Playwright live capture
    capture_signals.py  turns a capture into evidence (security block pages, OAuth, language, failures)
    html_*_signals.py, behavior_signals.py, host_path_reasoning.py, legitimacy_bundle.py, org_style_signals.py
    brand_coherence.py  page brand vs domain
    registries.py       trusted / platform / official domain lists
    domain_utils.py     shared host helpers
    verdict_rules.py    score adjustments before the judge (rescue, hosting trust, caps)
    eal.py              the Evidence Adjudication Layer (final verdict)
    verdict_policy.py   thresholds for the 3-way verdict
    frontend.py         Streamlit UI
```

Tests: `pytest`. `tests/test_golden_dashboard.py` replays 103 frozen cases through the dashboard and fails if any output changes; re-record only for intended changes (`tests/golden/record_golden.py`).
