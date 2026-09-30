# Verified metrics

Every number below was produced by running `phishguard evaluate` on the trained run. Nothing is copied from older reports. No synthetic data is used. Raw values: `metrics/results/evaluation.json`.

- Generated: 2026-09-30T08:14:14+00:00
- Git commit: `be39bd122e41c8bc3cc2cef4126d3ffcaea460d3`
- Code trees: src `883bb3c59f`, tests `6b038a0a67`, data/evaluation `075500398c`
- Seed: 42 (sampling, splits, models, bootstrap)
- Environment: Python 3.11.15, Linux 6.18.44-fc-v50, Intel(R) Xeon(R) Processor @ 2.10GHz (2 vCPU), none (CPU only)
- Packages: numpy 1.26.4, pandas 2.3.3, scikit-learn 1.5.1, xgboost 2.1.4, lightgbm 4.7.0, joblib 1.4.2, tldextract 5.3.2, pytest 8.4.2, playwright 1.49.0
- Reproduce: `phishguard train --full` then `phishguard evaluate --with-tests` (see `metrics/reproduce.sh`)

## Headline

| What | Value | Context |
|---|---|---|
| Primary model (XGBoost) held-out F1 | 0.801 | 95% CI 0.799 to 0.803; 147,702 test URLs from domains never seen in training; majority baseline F1 0.000 |
| ROC-AUC / PR-AUC | 0.880 / 0.849 | ROC-AUC 95% CI 0.878 to 0.881; baseline 0.500 / 0.464 |
| Precision / recall / false-positive rate | 0.772 / 0.832 / 21.3% | threshold 0.5 on raw model probability |
| Official brand URLs flagged by Layer 1 alone | 40.4% | 136 real official brand URLs (test half), app threshold |
| Official brand URLs the dashboard calls likely_phishing | 0.0% | ML-only dashboard path (no live capture); the rest are routed to uncertain |
| Real PhishStats phishing URLs flagged by Layer 1 | 81.0% | 284 URLs collected 2025-04 to 2026-04, hosts excluded from training |
| PhishStats URLs the dashboard calls likely_phishing | 24.3% | ML-only dashboard path |
| Tests | 365 / 365 passed | 54.0% line coverage of src/phishguard |

## Data

| What | Value |
|---|---|
| Source | Kaggle harisudhan411/phishing-and-legitimate-urls (`data/raw/kaggle/phishing_and_legitimate_urls.csv`, sha256 `330901bb4fbb63ff...`) |
| Raw rows x columns | 822,010 x 2 (427,028 legitimate, 394,982 phishing); no date column |
| After canonical dedupe | 794,563 |
| Run mode | FULL_DATASET; 823 curated legitimate homepages added |
| Removed so evaluation stays out-of-sample | 46,712 rows on legit-evaluation domains, 199 on phishing-evaluation hosts |
| Train / test rows | 600,743 / 147,702 (train 297,933 phishing, test 68,568 phishing) |
| Split | StratifiedGroupKFold by registered domain: 259,487 / 64,867 domains, overlap 0, rows without a parsable domain 820 |
| Fit / validation rows (inside train) | 511,749 / 88,994 (validation also grouped by domain) |
| Train/serve feature parity | 300 rows re-extracted the app's way, 0 mismatches |
| Model features | 54 |

## Held-out test (threshold 0.5)

| Model | Precision | Recall | F1 | F1 95% CI | ROC-AUC | PR-AUC | FPR | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | n/a | 0.500 | 0.464 | 0.0% | 79,134 / 0 / 68,568 / 0 |
| Logistic Regression | 0.679 | 0.801 | 0.735 | 0.732 to 0.737 | 0.773 | 0.697 | 32.8% | 53,161 / 25,973 / 13,675 / 54,893 |
| Random Forest | 0.784 | 0.802 | 0.793 | 0.791 to 0.795 | 0.884 | 0.866 | 19.2% | 63,966 / 15,168 / 13,585 / 54,983 |
| XGBoost (primary) | 0.772 | 0.832 | 0.801 | 0.799 to 0.803 | 0.880 | 0.849 | 21.3% | 62,253 / 16,881 / 11,499 / 57,069 |
| LightGBM | 0.764 | 0.847 | 0.803 | 0.801 to 0.806 | 0.877 | 0.833 | 22.7% | 61,203 / 17,931 / 10,502 / 58,066 |

Primary at the app's threshold (calibrated probability >= 0.5): precision 0.756, recall 0.859, F1 0.804, FPR 24.1%. Brier score raw 0.1408, calibrated 0.1455.

Selection: highest validation PR-AUC (domain-grouped validation split of the training rows); tie-break: fewer false alarms on the official-brand validation half. Validation PR-AUC per model: Logistic Regression 0.897, Random Forest 0.936, XGBoost 0.938, LightGBM 0.936.

## Cross-validation (StratifiedGroupKFold(5) by registered domain, 100,000 rows)

| Model | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|
| Majority-class baseline | 0.125 ± 0.280 | 0.500 ± 0.000 | 0.492 ± 0.025 | 20.0% ± 44.7 |
| Logistic Regression | 0.759 ± 0.018 | 0.832 ± 0.020 | 0.835 ± 0.027 | 28.5% ± 4.6 |
| Random Forest | 0.815 ± 0.013 | 0.902 ± 0.012 | 0.902 ± 0.014 | 16.6% ± 2.3 |
| XGBoost | 0.828 ± 0.015 | 0.908 ± 0.016 | 0.907 ± 0.019 | 18.3% ± 2.5 |
| LightGBM | 0.830 ± 0.013 | 0.909 ± 0.019 | 0.906 ± 0.022 | 19.0% ± 2.6 |

Mean ± standard deviation over 5 folds; domain overlap per fold: [0, 0, 0, 0, 0]. The majority-class baseline predicts whichever class is larger in each training fold, so in a fold where phishing is the majority it flags everything.

## External sets (never in training)

| Set | URLs | Truth | Primary flags (app threshold) | Each model flags (raw 0.5) | Overlap with training |
|---|---|---|---|---|---|
| official_brand_test_half | 136 | legitimate | 40.4% | Logistic Regression 30.1%, Random Forest 31.6%, XGBoost 36.8%, LightGBM 37.5% | 0 domain / 0 host |
| official_brand_validation_half | 162 | legitimate | 42.0% | Logistic Regression 31.5%, Random Forest 30.2%, XGBoost 37.0%, LightGBM 39.5% | 0 domain / 0 host |
| phishstats | 284 | phishing | 81.0% | Logistic Regression 74.6%, Random Forest 75.7%, XGBoost 78.9%, LightGBM 80.6% | 36 domain / 0 host |

For legitimate sets the flag rate is the false-alarm rate; for phishing sets it is recall. The overlap column counts URLs whose registered domain or exact host also appears in the training rows. The official-brand validation half was used to break model-selection ties, so quote the test half.

## Dashboard verdicts (ML-only (reinforcement=False): no live page capture)

| Set | URLs | likely_phishing | uncertain | likely_legitimate | Rate | Layer 1 flag rate |
|---|---|---|---|---|---|---|
| official_brand_test_half | 136 | 0 | 136 | 0 | false alarms 0.0% | 40.4% |
| phishstats | 284 | 69 | 215 | 0 | detected 24.3% | 81.0% |
| hard_legit | 15 | 0 | 15 | 0 | false alarms 0.0% | 86.7% |
| suite:obvious_legit | 6 | 0 | 6 | 0 | false alarms 0.0% | 100.0% |
| suite:tricky_legit | 6 | 0 | 6 | 0 | false alarms 0.0% | 100.0% |
| suite:obvious_phish | 2 | 2 | 0 | 0 | detected 100.0% | 100.0% |
| suite:hard_phishing | 4 | 4 | 0 | 0 | detected 100.0% | 100.0% |
| heldout_test_sample_legit | 200 | 18 | 182 | 0 | false alarms 9.0% | 26.0% |
| heldout_test_sample_phishing | 200 | 47 | 153 | 0 | detected 23.5% | 87.5% |

Without live capture the adjudication layer treats missing page evidence as a reason for `uncertain`, not as proof of safety.

## Latency

| Path | p50 | p95 | Context |
|---|---|---|---|
| Layer 1 only (features + model + calibration) | 14.6 ms | 20.7 ms | 400 held-out URLs; Intel(R) Xeon(R) Processor @ 2.10GHz; warm process, one URL at a time, includes 4-model agreement |
| Dashboard, ML-only (all rules + EAL) | 149.1 ms | 188.8 ms | same URLs |

## Full system with live page capture

From `phishguard evaluate-live` (GitHub Actions, commit `2b90195f56`, 2026-09-30T17:51:13+00:00). Each page was captured once with Playwright and analyzed twice (legitimacy rescue on and off). Dummy login interaction: on. Raw values: `metrics/results/live_capture.json`.

| Set | URLs | Captured OK | likely_phishing | uncertain | likely_legitimate | Pass rate | Pass rate (captured OK) | likely_phishing with rescue on / off |
|---|---|---|---|---|---|---|---|---|
| eal_edge_cases | 15 | 14 (93.3%) | 3 | 8 | 4 | 86.7% | 85.7% | 3 / 3 (0 changed) |
| url_suites | 18 | 14 (77.8%) | 7 | 4 | 7 | 94.4% | 92.9% | 7 / 7 (0 changed) |
| hard_legit | 15 | 15 (100.0%) | 2 | 6 | 7 | 86.7% | 86.7% | 2 / 2 (0 changed) |
| official_brand | 136 | 134 (98.5%) | 8 | 26 | 102 | 94.1% | 94.8% | 8 / 8 (0 changed) |
| phishstats_live | 0 | 0 (n/a) | 0 | 0 | 0 | n/a | n/a | 0 / 0 (0 changed) |
| tranco_live | 60 | 42 (70.0%) | 22 | 10 | 28 | 63.3% | 90.5% | 22 / 22 (0 changed) |

Pass means: legitimate sets not labeled likely_phishing; phishing sets labeled likely_phishing; the edge cases follow their own expected outcome (`data/evaluation/eal_edge_cases.json`). Live phishing pages are often already taken down, so check the captured-OK column before reading a phishing pass rate.

Full-path latency (capture + analysis): p50 7.1 s, p95 16.1 s over 244 URLs (capture (Playwright) + full analysis, one URL at a time, GitHub-hosted runner).

Edge cases that did not pass:

| URL | Expected | Verdict | Captured OK |
|---|---|---|---|
| `https://mrbslink.weebly.com/` | phishing | uncertain | True |
| `https://gghdgsyttetyeyy72.weebly.com/` | phishing | uncertain | True |


## Earlier baseline

The pre-rebuild audit (leaky split, scheme artifact, stale calibrator) is kept for comparison in `metrics/audit_baseline/` and at git tag `audit-baseline-2026-09-29`. Its numbers describe the old code and must not be quoted for the current system.
