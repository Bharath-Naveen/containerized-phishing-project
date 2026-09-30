# Verified metrics

Every number below was produced by running the code in this repo on the real Kaggle data. No synthetic data is used anywhere in this audit. Numbers from older reports or docs are not repeated here unless re-run.

- Generated: 2026-09-29T20:20:23+00:00
- Git commit: `9c04fdd46f98f407a04e7ce356279ce0389c1ddf` (code trees: src `d855aac065`, tests `dbe8341c37`, data/evaluation `15d94fe810`, requirements.txt `23f4eef21e`)
- Seed: 42 (numpy, sklearn, sampling, splits, bootstrap)
- Environment: Python 3.11.15, Linux 6.18.44-fc-v37, Intel(R) Xeon(R) Processor @ 2.10GHz (2 vCPU, 7.8 GB RAM), none (CPU only)
- Packages: numpy 1.26.4, pandas 2.3.3, scikit-learn 1.5.1, xgboost 2.1.4, lightgbm 4.7.0, joblib 1.4.2, tldextract 5.3.2, pytest 8.4.2, pytest-cov 7.1.0, playwright 1.49.0
- Data: `data/raw/kaggle/phishing_and_legitimate_urls.csv (Kaggle harisudhan411/phishing-and-legitimate-urls, not committed)`, sha256 `330901bb4fbb63ffc25d371cf0bed919c1ca031ea1c809bfc9a154a2d79e4a29`
- Reproduce everything: `bash metrics/reproduce.sh` (raw per-step outputs in `metrics/results/`)


## Tests

| Metric | Value | Context | Status |
|---|---|---|---|
| pytest collected / passed / failed / skipped | 253 / 253 / 0 / 0 | pass rate 100.0%; runtime 17.8 s | VERIFIED |
| Line coverage of src/ (pytest-cov) | 52.1% | 4500 of 8643 statements | VERIFIED |

## Data

| Metric | Value | Context | Status |
|---|---|---|---|
| Raw Kaggle rows x columns | 822,010 x 2 | legit (status 1) 427,028, phishing (status 0) 394,982; no date column | VERIFIED |
| Rows after URL canonical dedupe | 796,446 | from 822,010 | VERIFIED |
| Documented default sample and grouped split | 50,896 rows: train 41,403 / test 9,493 | 50,000 stratified (seed 42) + 896 curated legit URLs; StratifiedGroupKFold by registered domain; train/test domain overlap 0, URL overlap 0 | VERIFIED |
| Features used by the models | 56 | 59 extracted; has_https dropped on purpose, tld and public_suffix dropped as high-cardinality | VERIFIED |

## Runtime

| Metric | Value | Context | Status |
|---|---|---|---|
| Layer-1 pipeline end to end (ingest, dedupe, sample, enrich, split, train 4 models) | 87.9 s | 50K sample, CPU only (see environment) | VERIFIED |
| Layer-1 pipeline end to end, full data | 507.7 s | run_kaggle_pipeline(full_dataset=True) with checkpoint_every=10**9 (default checkpointing is what makes --full take hours) | VERIFIED |

## Layer-1 models (50K default run)

| Metric | Value | Context | Status |
|---|---|---|---|
| Held-out: dummy_most_frequent | P 0.000, R 0.000, F1 0.000, ROC-AUC 0.500, PR-AUC 0.473, FPR 0.0% | held-out test 9,493 rows (4,489 phishing), unseen domains, threshold 0.5; confusion [[tn,fp],[fn,tp]] = [[5004, 0], [4489, 0]] | VERIFIED |
| Held-out: logistic_regression | P 0.751, R 0.740, F1 0.745, ROC-AUC 0.851, PR-AUC 0.853, FPR 22.0% | held-out test 9,493 rows (4,489 phishing), unseen domains, threshold 0.5; F1 95% CI 0.734 to 0.756; ROC-AUC 95% CI 0.843 to 0.859; confusion [[tn,fp],[fn,tp]] = [[3902, 1102], [1169, 3320]] | VERIFIED |
| Held-out: random_forest | P 0.938, R 0.775, F1 0.849, ROC-AUC 0.924, PR-AUC 0.937, FPR 4.6% | held-out test 9,493 rows (4,489 phishing), unseen domains, threshold 0.5; F1 95% CI 0.840 to 0.857; ROC-AUC 95% CI 0.919 to 0.930; confusion [[tn,fp],[fn,tp]] = [[4774, 230], [1010, 3479]] | VERIFIED |
| Held-out: xgboost | P 0.928, R 0.796, F1 0.857, ROC-AUC 0.939, PR-AUC 0.947, FPR 5.6% | held-out test 9,493 rows (4,489 phishing), unseen domains, threshold 0.5; F1 95% CI 0.849 to 0.865; ROC-AUC 95% CI 0.934 to 0.944; confusion [[tn,fp],[fn,tp]] = [[4725, 279], [915, 3574]] | VERIFIED |
| Held-out: lightgbm | P 0.910, R 0.815, F1 0.860, ROC-AUC 0.944, PR-AUC 0.951, FPR 7.2% | held-out test 9,493 rows (4,489 phishing), unseen domains, threshold 0.5; F1 95% CI 0.852 to 0.868; ROC-AUC 95% CI 0.940 to 0.948; confusion [[tn,fp],[fn,tp]] = [[4642, 362], [829, 3660]] | VERIFIED |
| 5-fold grouped CV: dummy_most_frequent | F1 0.000 ± 0.000, ROC-AUC 0.500 ± 0.000, PR-AUC 0.455 ± 0.036 | StratifiedGroupKFold(5) by registered domain on all 50,896 rows; mean ± std across folds; group overlap per fold [0, 0, 0, 0, 0] | VERIFIED |
| 5-fold grouped CV: logistic_regression | F1 0.742 ± 0.012, ROC-AUC 0.843 ± 0.024, PR-AUC 0.824 ± 0.041 | StratifiedGroupKFold(5) by registered domain on all 50,896 rows; mean ± std across folds; group overlap per fold [0, 0, 0, 0, 0] | VERIFIED |
| 5-fold grouped CV: random_forest | F1 0.849 ± 0.003, ROC-AUC 0.931 ± 0.007, PR-AUC 0.938 ± 0.007 | StratifiedGroupKFold(5) by registered domain on all 50,896 rows; mean ± std across folds; group overlap per fold [0, 0, 0, 0, 0] | VERIFIED |
| 5-fold grouped CV: xgboost | F1 0.861 ± 0.002, ROC-AUC 0.941 ± 0.002, PR-AUC 0.945 ± 0.005 | StratifiedGroupKFold(5) by registered domain on all 50,896 rows; mean ± std across folds; group overlap per fold [0, 0, 0, 0, 0] | VERIFIED |
| 5-fold grouped CV: lightgbm | F1 0.866 ± 0.003, ROC-AUC 0.946 ± 0.004, PR-AUC 0.949 ± 0.005 | StratifiedGroupKFold(5) by registered domain on all 50,896 rows; mean ± std across folds; group overlap per fold [0, 0, 0, 0, 0] | VERIFIED |
| Composite-rule winner | logistic_regression | rule: argmax(F1 - mean raw P(phish) on 8 official HTTPS URLs) (src/pipeline/train.py); best F1 = lightgbm, best ROC-AUC = lightgbm | VERIFIED |
| Winner at the app threshold (calibrated P >= 0.5): logistic_regression | P 0.800, R 0.658, F1 0.722, FPR 14.7% | 3-way split from the proxy formula in benchmark_url_suites.py (0.65 x calibrated, thresholds 0.56/0.38), not the full dashboard: {'phishing_rows': {'likely_phishing': 1717, 'uncertain': 959, 'likely_legitimate': 1813}, 'legit_rows': {'likely_phishing': 82, 'uncertain': 442, 'likely_legitimate': 4480}} | VERIFIED |

## Full-data run (~797K rows)

| Metric | Value | Context | Status |
|---|---|---|---|
| Rows and grouped split | train 657,226 / test 140,057 | all 796,446 deduplicated Kaggle rows + 837 curated legit; 259,504 / 64,902 domain groups; overlap 0 | VERIFIED |
| Held-out: dummy_most_frequent | P 0.000, R 0.000, F1 0.000, ROC-AUC 0.500, PR-AUC 0.505, FPR 0.0% | test 140,057 rows, unseen domains, threshold 0.5 | VERIFIED |
| Held-out: logistic_regression | P 0.772, R 0.776, F1 0.774, ROC-AUC 0.862, PR-AUC 0.881, FPR 23.4% | test 140,057 rows, unseen domains, threshold 0.5; F1 95% CI 0.772 to 0.776; ROC-AUC 95% CI 0.860 to 0.864 | VERIFIED |
| Held-out: random_forest | P 0.936, R 0.803, F1 0.864, ROC-AUC 0.940, PR-AUC 0.953, FPR 5.6% | test 140,057 rows, unseen domains, threshold 0.5; F1 95% CI 0.862 to 0.866; ROC-AUC 95% CI 0.939 to 0.941 | VERIFIED |
| Held-out: xgboost | P 0.939, R 0.822, F1 0.876, ROC-AUC 0.950, PR-AUC 0.960, FPR 5.5% | test 140,057 rows, unseen domains, threshold 0.5; F1 95% CI 0.875 to 0.878; ROC-AUC 95% CI 0.949 to 0.951 | VERIFIED |
| Held-out: lightgbm | P 0.921, R 0.844, F1 0.881, ROC-AUC 0.954, PR-AUC 0.963, FPR 7.3% | test 140,057 rows, unseen domains, threshold 0.5; F1 95% CI 0.879 to 0.883; ROC-AUC 95% CI 0.953 to 0.955 | VERIFIED |
| Composite-rule winner / best F1 | logistic_regression / lightgbm | composite = argmax(F1 - mean P(phish) on 8 official HTTPS URLs) | VERIFIED |
| External sets: logistic_regression | official-brand FP 54.4% (162/298); PhishStats recall 91.5% (n=284) | Layer-1 only, raw P >= 0.5; 219 official and 56 PhishStats URLs share a domain with training rows | VERIFIED |
| External sets: random_forest | official-brand FP 35.2% (105/298); PhishStats recall 86.6% (n=284) | Layer-1 only, raw P >= 0.5; 219 official and 56 PhishStats URLs share a domain with training rows | VERIFIED |
| External sets: xgboost | official-brand FP 41.9% (125/298); PhishStats recall 89.8% (n=284) | Layer-1 only, raw P >= 0.5; 219 official and 56 PhishStats URLs share a domain with training rows | VERIFIED |
| External sets: lightgbm | official-brand FP 44.3% (132/298); PhishStats recall 88.4% (n=284) | Layer-1 only, raw P >= 0.5; 219 official and 56 PhishStats URLs share a domain with training rows | VERIFIED |

## Ablations

| Metric | Value | Context | Status |
|---|---|---|---|
| Scheme-artifact check: logistic_regression | F1 0.745 -> 0.747; ROC-AUC 0.851 -> 0.838 | same rows and split, features recomputed with every URL rewritten to https:// | VERIFIED |
| Scheme-artifact check: random_forest | F1 0.849 -> 0.820; ROC-AUC 0.924 -> 0.911 | same rows and split, features recomputed with every URL rewritten to https:// | VERIFIED |
| Scheme-artifact check: xgboost | F1 0.857 -> 0.827; ROC-AUC 0.939 -> 0.926 | same rows and split, features recomputed with every URL rewritten to https:// | VERIFIED |
| Scheme-artifact check: lightgbm | F1 0.860 -> 0.837; ROC-AUC 0.944 -> 0.930 | same rows and split, features recomputed with every URL rewritten to https:// | VERIFIED |
| Evaluation-list domains found in training rows | 1,414 train rows | by source {'kaggle_harisudhan411_phishing_and_legitimate_urls': 1355, 'curated_simple_legit': 59}; 18 domains from url_suites, hard_legit, composite audit URLs | VERIFIED |
| Composite winner: as-is / scheme-neutral / decontaminated | logistic_regression / logistic_regression / logistic_regression | best F1 in each: lightgbm / lightgbm / lightgbm | VERIFIED |
| Legitimacy rescue FP delta with live capture | NOT RUN | rescue conditions depend on redirect chain, form targets and DOM risk from Playwright capture; no internet egress here | NOT RUN |
| EAL edge-case validation (15 live URLs in outputs/reports/eal_edge_case_validation.md) | NOT RUN | needs live Playwright capture of real sites; no internet egress in the verification sandbox | NOT RUN |

## Deployed models (500K run, shipped in outputs/models)

| Metric | Value | Context | Status |
|---|---|---|---|
| Held-out: dummy_most_frequent | P 0.000, R 0.000, F1 0.000, ROC-AUC 0.500, PR-AUC 0.456, FPR 0.0% | saved split of run 20260427_234728: train 401,918 / test 98,082, 14,823 registered domains appear in both (see the split and feature fix section); threshold 0.5 | VERIFIED |
| Held-out: random_forest (deployed layer1_primary) | P 0.825, R 0.824, F1 0.825, ROC-AUC 0.917, PR-AUC 0.915, FPR 14.7% | saved split of run 20260427_234728: train 401,918 / test 98,082, 14,823 registered domains appear in both (see the split and feature fix section); threshold 0.5; F1 95% CI 0.822 to 0.827 | VERIFIED |
| Held-out: logistic_regression (witness) | P 0.782, R 0.793, F1 0.788, ROC-AUC 0.871, PR-AUC 0.862, FPR 18.6% | saved split of run 20260427_234728: train 401,918 / test 98,082, 14,823 registered domains appear in both (see the split and feature fix section); threshold 0.5; F1 95% CI 0.785 to 0.790 | VERIFIED |
| Held-out: xgboost (witness) | P 0.839, R 0.765, F1 0.800, ROC-AUC 0.905, PR-AUC 0.903, FPR 12.3% | saved split of run 20260427_234728: train 401,918 / test 98,082, 14,823 registered domains appear in both (see the split and feature fix section); threshold 0.5; F1 95% CI 0.797 to 0.803 | VERIFIED |
| Held-out: lightgbm (witness) | P 0.839, R 0.837, F1 0.838, ROC-AUC 0.928, PR-AUC 0.925, FPR 13.4% | saved split of run 20260427_234728: train 401,918 / test 98,082, 14,823 registered domains appear in both (see the split and feature fix section); threshold 0.5; F1 95% CI 0.836 to 0.841 | VERIFIED |
| Layer-1 FP rate on curated official brand URLs | 29.9% (89 of 298) | phishing_dataset/dataset_test_recent.csv brand_official rows, collected 2026-04-26; 1 URL(s) in training; raw P >= 0.5 | VERIFIED |
| Layer-1 recall on PhishStats phishing URLs | 94.7% (n=284) | URLs not in the deployed training split; 61 share a registered domain with training rows; collected ['2025-04-29', '2026-04-26'] | VERIFIED |
| Brier score: raw / shipped calibrator / refit calibrator | 0.1149 / 0.1168 / 0.1129 | deploy_layer1_primary.py copies the model only; the calibrator on disk was fit for a different model (file dated before the RF was deployed). | VERIFIED |

## Deployed models: split and feature fix (evaluation only)

| Metric | Value | Context | Status |
|---|---|---|---|
| Test rows that share a real registered domain with training rows | 54,358 of 98,082 (55.4%) | retrain_with_fresh.py kept scheme-less URLs, so 301,164 of 401,918 train rows got a per-URL 'malformed::' group key instead of their domain | VERIFIED |
| random_forest (deployed layer1_primary): as reported -> leak-free rows -> leak-free rows with app features | ROC-AUC 0.917 -> 0.884 -> 0.866; FPR 14.7% -> 35.4% -> 75.2%; recall 82.4% -> 88.1% -> 98.3% | leak-free rows: 43,724 (29,915 phishing, 13,809 legit, so F1 is inflated by prevalence; use ROC-AUC and FPR) | VERIFIED |
| logistic_regression (witness): as reported -> leak-free rows -> leak-free rows with app features | ROC-AUC 0.871 -> 0.836 -> 0.723; FPR 18.6% -> 39.5% -> 77.0%; recall 79.3% -> 86.4% -> 96.1% | leak-free rows: 43,724 (29,915 phishing, 13,809 legit, so F1 is inflated by prevalence; use ROC-AUC and FPR) | VERIFIED |
| xgboost (witness): as reported -> leak-free rows -> leak-free rows with app features | ROC-AUC 0.905 -> 0.882 -> 0.847; FPR 12.3% -> 26.4% -> 71.3%; recall 76.5% -> 83.0% -> 97.6% | leak-free rows: 43,724 (29,915 phishing, 13,809 legit, so F1 is inflated by prevalence; use ROC-AUC and FPR) | VERIFIED |
| lightgbm (witness): as reported -> leak-free rows -> leak-free rows with app features | ROC-AUC 0.928 -> 0.895 -> 0.855; FPR 13.4% -> 32.0% -> 68.2%; recall 83.7% -> 88.4% -> 97.5% | leak-free rows: 43,724 (29,915 phishing, 13,809 legit, so F1 is inflated by prevalence; use ROC-AUC and FPR) | VERIFIED |

## Dashboard verdicts, ML-only path (deployed)

| Metric | Value | Context | Status |
|---|---|---|---|
| hard_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=15) | verdicts {'uncertain': 15}; Layer-1 flag rate 13.3% | VERIFIED |
| kaggle_heldout_sample:legit | FP 58.0%; not-flagged 42%; strict likely_legitimate 0% (n=200) | verdicts {'likely_phishing': 116, 'uncertain': 84}; Layer-1 flag rate 67.0% | VERIFIED |
| kaggle_heldout_sample:phishing | 70% likely_phishing (n=200) | verdicts {'likely_phishing': 141, 'uncertain': 59}; Layer-1 flag rate 98.5% | VERIFIED |
| official_brand_2026_04_26 | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=298) | verdicts {'uncertain': 298}; Layer-1 flag rate 29.5% | VERIFIED |
| simple_legit | FP 0.6%; not-flagged 99%; strict likely_legitimate 0% (n=884) | verdicts {'uncertain': 879, 'likely_phishing': 5}; Layer-1 flag rate 93.2% | VERIFIED |
| suite:hard_phishing | 75% likely_phishing (n=4) | verdicts {'likely_phishing': 3, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:obvious_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 66.7% | VERIFIED |
| suite:obvious_phish | 50% likely_phishing (n=2) | verdicts {'likely_phishing': 1, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:tricky_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 16.7% | VERIFIED |
| Legitimacy rescue ON vs OFF | legit FPs 116 (on) vs 116 (off); verdicts changed 0 of 731 | PHISH_LEGITIMACY_RESCUE_ENABLED toggled; no live capture, so capture-dependent rescue conditions cannot fire | VERIFIED |

## Dashboard verdicts, ML-only path (asis_50k)

| Metric | Value | Context | Status |
|---|---|---|---|
| hard_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=15) | verdicts {'uncertain': 15}; Layer-1 flag rate 6.7% | VERIFIED |
| kaggle_heldout_sample:legit | FP 1.0%; not-flagged 99%; strict likely_legitimate 0% (n=200) | verdicts {'uncertain': 198, 'likely_phishing': 2}; Layer-1 flag rate 14.5% | VERIFIED |
| kaggle_heldout_sample:phishing | 18% likely_phishing (n=200) | verdicts {'uncertain': 163, 'likely_phishing': 37}; Layer-1 flag rate 66.5% | VERIFIED |
| official_brand_2026_04_26 | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=298) | verdicts {'uncertain': 298}; Layer-1 flag rate 31.2% | VERIFIED |
| simple_legit | FP 0.6%; not-flagged 99%; strict likely_legitimate 0% (n=884) | verdicts {'uncertain': 879, 'likely_phishing': 5}; Layer-1 flag rate 94.1% | VERIFIED |
| suite:hard_phishing | 75% likely_phishing (n=4) | verdicts {'likely_phishing': 3, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:obvious_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 66.7% | VERIFIED |
| suite:obvious_phish | 50% likely_phishing (n=2) | verdicts {'likely_phishing': 1, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:tricky_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 0.0% | VERIFIED |
| Legitimacy rescue ON vs OFF | legit FPs 2 (on) vs 2 (off); verdicts changed 0 of 731 | PHISH_LEGITIMACY_RESCUE_ENABLED toggled; no live capture, so capture-dependent rescue conditions cannot fire | VERIFIED |

## Dashboard verdicts, ML-only path (decontaminated)

| Metric | Value | Context | Status |
|---|---|---|---|
| hard_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=15) | verdicts {'uncertain': 15}; Layer-1 flag rate 26.7% | VERIFIED |
| kaggle_heldout_sample:legit | FP 1.0%; not-flagged 99%; strict likely_legitimate 0% (n=200) | verdicts {'uncertain': 198, 'likely_phishing': 2}; Layer-1 flag rate 14.0% | VERIFIED |
| kaggle_heldout_sample:phishing | 18% likely_phishing (n=200) | verdicts {'uncertain': 163, 'likely_phishing': 37}; Layer-1 flag rate 67.5% | VERIFIED |
| official_brand_2026_04_26 | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=298) | verdicts {'uncertain': 298}; Layer-1 flag rate 34.9% | VERIFIED |
| simple_legit | FP 0.6%; not-flagged 99%; strict likely_legitimate 0% (n=884) | verdicts {'uncertain': 879, 'likely_phishing': 5}; Layer-1 flag rate 96.5% | VERIFIED |
| suite:hard_phishing | 75% likely_phishing (n=4) | verdicts {'likely_phishing': 3, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:obvious_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 66.7% | VERIFIED |
| suite:obvious_phish | 50% likely_phishing (n=2) | verdicts {'likely_phishing': 1, 'uncertain': 1}; Layer-1 flag rate 100.0% | VERIFIED |
| suite:tricky_legit | FP 0.0%; not-flagged 100%; strict likely_legitimate 0% (n=6) | verdicts {'uncertain': 6}; Layer-1 flag rate 33.3% | VERIFIED |
| Legitimacy rescue ON vs OFF | legit FPs 2 (on) vs 2 (off); verdicts changed 0 of 731 | PHISH_LEGITIMACY_RESCUE_ENABLED toggled; no live capture, so capture-dependent rescue conditions cannot fire | VERIFIED |

## Latency

| Metric | Value | Context | Status |
|---|---|---|---|
| Layer-1 only, per URL (deployed) | p50 101.7 ms, p95 129.8 ms | n=400 held-out URLs, warm process, CPU only | VERIFIED |
| Dashboard ML-only path incl. EAL, per URL (deployed) | p50 228.1 ms, p95 280.6 ms | n=400, reinforcement=False, includes 4-model agreement | VERIFIED |
| Layer-1 only, per URL (asis_50k) | p50 8.8 ms, p95 12.7 ms | n=400 held-out URLs, warm process, CPU only | VERIFIED |
| Dashboard ML-only path incl. EAL, per URL (asis_50k) | p50 120.0 ms, p95 150.0 ms | n=400, reinforcement=False, includes 4-model agreement | VERIFIED |
| Layer-1 only, per URL (decontaminated) | p50 8.7 ms, p95 12.5 ms | n=400 held-out URLs, warm process, CPU only | VERIFIED |
| Dashboard ML-only path incl. EAL, per URL (decontaminated) | p50 116.7 ms, p95 148.1 ms | n=400, reinforcement=False, includes 4-model agreement | VERIFIED |
| Full path with Playwright live capture | NOT RUN | verification sandbox has no general internet egress (only package registries); needs a machine with internet | NOT RUN |

## How to read this

- The audit ran on a clean clone of GitHub `main`; its `src/`, `tests/`, `data/evaluation/` and `requirements.txt` trees are byte-identical to the local commit `4cb3d6b` (that commit only adds docs/BUILD_STORY.md). The code-tree hashes above let you check this with `git rev-parse HEAD:src`.
- 50K default run: the documented default (`python -m src.pipeline.run_kaggle_pipeline`), Kaggle only, re-run from scratch here.
- Deployed models: the four `.joblib` files the app loads today (500K-row run 20260427_234728, byte-identical to that run's folder). They were re-scored on that run's saved held-out split, not retrained.
- Dashboard verdict rows use the ML-only path (`reinforcement=False`). Live Playwright capture could not run in the verification sandbox.
- Several curated evaluation URLs are also added to training by `simple_legit_augment` (see Ablations). Treat suite and hard-legit pass rates for the as-is and deployed models as in-sample.
