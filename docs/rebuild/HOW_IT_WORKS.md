# How the project works (current state, after rebuild Phase 1)

A plain-language map of the system. It is updated as the rebuild changes things; the dated history of each change is in [REBUILD_LOG.md](REBUILD_LOG.md).

## The one-paragraph version

You give it a URL. A fast machine-learning model scores the URL text alone (Layer 1). Optionally, a headless browser opens the page and collects evidence: redirects, forms, TLS state, network requests (Layer 2). The HTML, the host and path, and brand-to-domain fit are then analyzed (Layer 3 and brand context). Finally a rule-based judge, the Evidence Adjudication Layer (EAL), weighs everything and returns `likely_phishing`, `uncertain` or `likely_legitimate`, with reasons. The ML score alone never decides; the EAL does.

## Part 1: How a URL is scored (runtime, `src/app_v1/`)

| Step | What happens | Where |
|---|---|---|
| 1. Normalize | The URL is put in one canonical form (lowercase host, scheme added if missing, trailing slash rules). Features are computed from a scheme-neutral copy (`http://` forced), so `https://x.com` and `x.com` look identical to the model. | `src/pipeline/url_normalize.py` |
| 2. Layer 1 features | 59 cheap features from the URL text only: lengths, dots, digits, entropy, suspicious words, brand tokens in the wrong place, free-hosting domains, official-domain registry match. No network. 54 of them are used by the model (3 scheme-derived ones and 2 high-cardinality text columns are left out). | `src/pipeline/layer1_features.py`, `src/pipeline/features/` |
| 3. Layer 1 model | The **model bundle** (`outputs/models/layer1_bundle.joblib`) holds the trained pipeline, its own probability calibrator and the exact feature list. Output: raw and calibrated P(phishing). | `src/app_v1/ml_layer1.py` |
| 4. Model agreement | Four "witness" models (LR, RF, XGBoost, LightGBM) also score the URL; their votes become a consensus signal (strong_phishing, split, and so on). Supporting evidence only. | `ml_layer1.compute_layer1_model_agreement` |
| 5. Layer 2 capture (optional) | Playwright loads the page: final URL, redirect chain, form targets, TLS/cert state, network requests, HTML. Skipped in "ML-only" mode. | `src/app_v1/capture.py` |
| 6. Layer 3 analysis | HTML structure and DOM anomalies (login harvesters, wrappers), JS/network behavior, host/path reasoning (is this host shape suspicious? does the path fit?). | `html_*_signals.py`, `behavior_signals.py`, `host_path_reasoning.py` |
| 7. Brand and trust context | Does the page's brand match the domain? Is the domain in the small official-domain registry (a weak prior, not a whitelist)? Is it user-hosted (github.io, netlify.app)? | `analyze_dashboard.py`, `data/official_domains.json`, `data/reference/*.csv` |
| 8. EAL verdict | Adds up phishing, legitimacy and ambiguity signals, applies hard blockers (for example credential form posting to another domain), and picks the verdict. When evidence is missing or conflicting it says `uncertain` on purpose. | `analyze_dashboard._apply_evidence_adjudication_layer` |

Important current behavior: without Layer 2 capture, the EAL almost never says `likely_legitimate` (missing evidence is not treated as proof of safety). That is by design, and it is why the demo and metrics distinguish "ML-only" from "full capture".

## Part 2: How the model is trained (`src/pipeline/`)

Entry point: `python -m src.pipeline.run_kaggle_pipeline` (default 50,000-row sample; `--full` for all rows).

1. **Ingest** the Kaggle CSV (`url,status`; Kaggle `status 1` = legit is mapped to internal `label 0` = legit, `1` = phishing).
2. **Clean**: canonicalize every URL and drop exact duplicates (822,010 rows become about 796K).
3. **Sample**: stratified sample (keeps the class balance), seed 42.
4. **Augment**: add ~880 curated legitimate homepages (`data/evaluation/simple_legit_urls.jsonl`).
5. **Remove evaluation URLs** (new): any row on the registered domain of a legitimate evaluation URL, or on the exact host of a phishing evaluation URL, is dropped, so evaluation stays out-of-sample.
6. **Enrich**: compute the Layer 1 features for every row.
7. **Split**: train/test by registered domain (StratifiedGroupKFold), so a website is never in both. Guards stop the run if too many rows lack a real domain or if any domain overlaps.
8. **Train**: four models. Inside the training rows, a second domain-grouped split makes a validation set. A parity guard re-computes features for 300 stored rows the way the app does and stops the run on any mismatch.
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
