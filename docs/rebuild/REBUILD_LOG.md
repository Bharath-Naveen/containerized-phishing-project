# Rebuild log

What was changed, why, and what it did to the numbers. Newest entries at the bottom. For how the system works as a whole, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md). The plan these steps follow lives in the claude.ai project (`claude/rebuild-plan.md`).

Branch: `rebuild/verified-metrics`. Nothing here is on `main` until the pull request is merged.

---

## 2026-09-29 · Phase 0: Git cleanup and the "before" snapshot

**Why.** Your laptop copy and GitHub had each gained one commit the other did not have (BUILD_STORY.md locally, a README line on GitHub). And on Windows, almost every file showed as "modified" only because of line endings.

**What was done.**
- Started the branch from GitHub's `main` and replayed your BUILD_STORY.md commit on top, so both commits are kept.
- `ce9a78b` Repo hygiene: added `.gitattributes` so Git stores line endings consistently (the "every file modified" noise goes away). Started tracking `data/reference/*.csv`: two tiny public domain lists the app and 6 tests need; a fresh clone failed without them.
- `0ea50d9` Added the audit (`metrics/`) exactly as it was run, as the "before" picture. Tagged `audit-baseline-2026-09-29`.

**What it means for you.** Any later number can be compared against this tag.

---

## 2026-09-29 · Phase 1: Correctness fixes

Each fix below says what was wrong, how it works now, and where the code is. All 266 tests pass (253 original + 13 new). On a fresh clone with no trained models: 263 pass, 3 skip (model smoke tests), 0 fail; before this phase a fresh clone had 6 failures.

### 1. One way to read a URL (the root cause of the leak and the skew)

**Was wrong.** The Kaggle file mixes `https://x.com/a` with bare `x.com/a`. One training path (`retrain_with_fresh.py`, which produced the currently shipped models) passed bare hosts straight into feature extraction. Python's URL parser then saw *no hostname*, so hostname length, dot count, entropy and so on were all zero for 75% of rows. At runtime the app adds `http://` first, so the same URL produced different features in production than in training.

**Now.** `src/pipeline/url_normalize.py` is the single place that decides what a URL is. `extract_layer1_features()` normalizes its input itself, so it no longer matters who calls it or in what form. `retrain_with_fresh.py` canonicalizes instead of copying raw strings. Canonical form also treats `x.com` and `x.com/` as the same address.

### 2. The split really groups by website now

**Was wrong.** Grouping used the registered domain, but for bare hosts the parser failed and every row got its own random `malformed::` group. So 55% of the shipped models' "held-out" test rows were on websites also in training.

**Now.** `leak_safe_group_key()` canonicalizes before extracting the domain. Two **guards** (`src/pipeline/guards.py`) stop the pipeline with a plain message if more than 0.5% of rows lack a real domain, or if any domain is in both train and test. In the Phase 1 check run: 63 of 47,852 rows (0.13%) lacked a domain, 0 domains overlapped.

### 3. A train/serve parity check

**Now.** Before training, 300 stored training rows have their features recomputed exactly as the app would. Any difference stops the run. Phase 1 check: 300 checked, 0 mismatched. A unit test also checks that `x.com`, `http://x.com/`, `HTTPS://X.com` all produce identical model features.

### 4. Evaluation URLs stay out of training

**Was wrong.** The curated test URLs (for example google.com, wikipedia.org, the 15 "hard legit" URLs, and the 8 URLs the old model-selection rule scored) were being added to the training data. So some "tests" were questions the model had already seen the answers to.

**Now.** `drop_evaluation_rows()` (`src/pipeline/evaluation_sets.py`) removes every training row on the domain of a legitimate evaluation URL, or on the exact host of a phishing evaluation URL (host, not domain, because many phishing URLs live on shared hosts like github.io; dropping all of github.io would remove thousands of useful rows). The hard-legit list is no longer added to training at all. Two new evaluation files were created from your existing data: 298 official brand URLs (split by domain into a validation half and a test half) and 284 PhishStats phishing URLs. Phase 1 check: from the 50K sample, 3,017 rows on legitimate-evaluation domains and 13 rows on phishing-evaluation hosts were removed.

### 5. The model can no longer see http vs https

**Was wrong.** In this dataset, an explicit `https://` is far more common on phishing rows than legit ones (on the real web, nearly everything is https). The old code dropped the `has_https` column but kept two columns built from it, plus `url_length`, which is longer by one character for https.

**Now.** Features are computed on a copy of the URL with the scheme forced to `http://`, and all three scheme-derived columns are excluded from training. `has_https` still reports the real scheme for the dashboard. Model features: 56 before, 54 now.

### 6. Model and calibrator travel together (the "bundle")

**Was wrong.** Deploying a new model copied only the model file. The probability calibrator on disk was fit for an older, different model, so it made the shipped model's probabilities slightly worse, not better (Brier 0.1149 raw vs 0.1168 "calibrated").

**Now.** Training writes `layer1_bundle.joblib`: the model, its own calibrator, the exact feature list, and provenance (train-data hash, row counts, seed, date, selection rule). The app loads the bundle first and only falls back to the old loose files if no bundle exists. Deploying copies the bundle, and removes a stale one if the new run has none.

### 7. A model-selection rule that uses validation data

**Was wrong.** The "composite" rule scored each model as F1 minus its average phishing probability on 8 famous websites. Those 8 URLs were in the training data, and the rule picked Logistic Regression (the weakest model) every time.

**Now.** The default rule is `validated`: best PR-AUC on a validation set that is split off the training rows by domain, ties broken by fewer false alarms on the validation half of the official-brand URLs. The test set is never used to choose. Also fixed: the validation split used to be a random row split (so the calibrator saw the same websites it was trained on); it is now grouped by domain. The old rules are still available as options. Phase 1 check picked LightGBM.

### 8. Tests no longer touch your real files

**Was wrong.** Running `pytest` wrote reports into `outputs/reports/` and CSVs into `data/processed/` (your `split_leak_safe_stats.json` there was test output).

**Now.** `tests/conftest.py` points every test run at a temporary folder and copies the shipped models there so the model tests still run.

### 9. Obvious phishing no longer comes back as "uncertain"

**Was wrong.** `google-login-secure.xyz/signin` and `totally-fake-bank-login.xyz/verify` had raw ML scores of about 0.97, but without live capture the EAL returned `uncertain`. The host checker only looked for brand names in *subdomains*, so a brand inside the registered name itself (`google-login-secure`) passed as an ordinary host.

**Now.** `host_path_reasoning.py` also checks the registered name: a brand token used as a prefix or glued to other text on a non-official domain, or two or more credential-lure words (login, verify, bank, secure, account, ...) in the name. Either marks the host as a suspicious pattern, and the existing EAL rule then convicts when ML is very confident. How often the new rule fires on a 40,000-row Kaggle sample: 190 phishing vs 36 legit rows (about 84% phishing); 0 of 298 official brand URLs; 1 of 884 curated homepages. It only changes a verdict when the ML score is also high. Two regression tests pin this.

### Phase 1 check: what the numbers look like now

Command: `python metrics/scripts/10_phase1_check.py` (50K sample, seed 42, 73 s). Result file: `metrics/results/10_phase1_check.json`.

| | Audit baseline (before) | After Phase 1 |
|---|---|---|
| Rows in the 50K run after exclusions | 50,896 | 47,852 |
| Model features | 56 | 54 |
| Selected primary model | Logistic Regression (composite rule) | LightGBM (validated rule) |
| LightGBM test F1 / ROC-AUC | 0.860 / 0.944 | 0.823 / 0.908 |
| LightGBM test precision / recall | 0.910 / 0.815 | 0.805 / 0.842 |
| Layer 1 false alarms on official brand URLs (test half) | not measured this way | 33.8% (46 of 136) |
| Dashboard verdicts on those URLs (ML-only) | all `uncertain` | all `uncertain` (0 `likely_phishing`) |
| Layer 1 recall on 284 PhishStats phishing URLs | n/a for this run | 79.6% |
| URL suites, dashboard ML-only | obvious phish 1/2, hard phish 3/4 | obvious phish 2/2, hard phish 4/4 |

**Why the scores went down, and why that is good.** The before and after test sets are not identical (the evaluation domains were removed), and three shortcuts the model used to rely on are gone: the http/https artifact, easy evaluation domains in training, and a validation set that shared websites with training. The new numbers are what the model can actually do on websites it has never seen. The audit's scheme-only ablation predicted about 2 F1 points from the scheme fix alone; the rest comes from the harder, cleaner test set and grouped validation.

**What is still weak (for later phases).** URL-only models still flag about a third of real official brand URLs; the EAL keeps them at `uncertain`, so no false phishing verdict reaches the user, but the ML layer needs better legitimate training data (Tranco top sites were planned). That is a Phase 3 data task.

### Still open after Phase 1
- The shipped models in `outputs/models/` are the old, flawed ones. They get replaced after the full retrain in Phase 3.
- The `metrics/` audit scripts describe the *old* pipeline on purpose (they are the "before" record). Phase 3 replaces them with one evaluation command for the new code.
- Live-capture measurements wait for GitHub Actions (Phase 4).

---

## 2026-09-30 · Phase 2: Restructure (same behavior, clearer code)

**Goal.** Make the code easy to find your way around and easy to explain, without changing a single verdict. Current layout and a code map: [HOW_IT_WORKS.md](HOW_IT_WORKS.md), Part 5.

### Step 1: A safety net first (golden-output tests)

**What.** Before moving anything, the full dashboard output was frozen for 103 cases: 95 URLs through the ML-only path (the suites, hard-legit list, official brand and PhishStats samples, and tricky shapes like punycode, IP hosts, user-hosted clones) and 8 hand-built live-capture cases (legit same-domain login, credential form posting to another domain, wrapper page, blocked capture, news article, free-hosted brand clone, security block page, official authwall). The ML scores are recorded once and replayed, so the test protects the rules and the adjudication logic, not whichever model is on disk.

**Proof it works.** Nudging one verdict threshold from 0.56 to 0.57 made 52 of the 104 golden checks fail. So "all golden tests pass" really does mean "same outputs".

**It already caught a bug.** The trusted-domain registry cache ignored which file it was loaded from, so whichever test loaded it first decided what every later lookup saw. The golden test failed only when run after certain other tests. The cache is now keyed by file path (the platform registry already had this fix).

### Step 2: Archive what nothing uses

An import-graph scan found 19 modules that nothing in the app, the training pipelines or the audits reaches. They moved to `archive/legacy/` with a README each:
- `ai_adjudication/`: the retired OpenAI adjudication path and the old screenshot/compare triage flow built around it (9 app modules, 2 pipeline modules, their tests).
- `old_pipeline/`: 8 superseded pipeline scripts (old ingest, the non-grouped split, run_all, and so on).

`openai` was removed from `requirements.txt` since no running code imports it.

### Step 3: Move everything into one package

`src/pipeline/` and `src/app_v1/` became `src/phishguard/` with sub-packages by job: `urls`, `data`, `features`, `models`, `pipelines`, `evaluation`, `app`. Done with `git mv` (history kept) plus an automatic import rewrite. A few files got clearer names, for example `analyze_dashboard.py` is now `app/dashboard.py`, `run_kaggle_pipeline.py` is `pipelines/kaggle.py`, `split_leak_safe.py` is `data/split.py`. Imports are now `phishguard.*`, with `src/` on the path.

### Step 4: Split the 3,585-line dashboard file

It is now seven files, each with one job: `dashboard.py` (wires the layers together, 594 lines), `eal.py` (the judge), `verdict_rules.py` (adjustments before the judge), `capture_signals.py`, `registries.py`, `brand_coherence.py`, `domain_utils.py`. Function bodies were moved by a script, not retyped, and `dashboard.py` re-exports the moved names so old imports still work. Golden tests: identical.

While checking for undefined names, a latent bug surfaced: the full (non-Layer-1) enrich path in `data/enrich.py` called `safe_hostname` without importing it, so it would have crashed the first time anyone used it. Fixed.

### Step 5: One command, one config

- `phishguard train | analyze | evaluate | deploy | serve` (install with `pip install -e .`; `python -m phishguard` also works).
- `phishguard/config.py` holds the seed (42) and defaults.
- Docker: `PYTHONPATH=/app/src`, new frontend path, and the image now includes the small registries the app needs (`data/official_domains.json`, `data/reference/`, `data/evaluation/`). A plain `docker build` used to leave them out, so a hosted container would have run without them. Not built here because this workspace has no Docker daemon; GitHub Actions builds it in Phase 4.
- README (setup, commands, layout) and docs updated to the new paths.

### Gate 2 result

| Check | Result |
|---|---|
| Full test suite | 362 passed (359 carried over + 3 new CLI tests; 11 AI-only tests moved to the archive with their code) |
| Golden outputs (103 cases) | identical before and after every step |
| Undefined names (ruff F821) in `src/phishguard` | 0 |

### Still open after Phase 2
- `phishguard evaluate` currently runs the URL-suite benchmark only; Phase 3 turns it into the single reproducible evaluation command.
- The `metrics/` audit scripts now import the new package, so running them on this branch measures the new code. The "before" numbers stay pinned to tag `audit-baseline-2026-09-29`.

---

## 2026-09-30 · Phase 3: Retrain and evaluate

### Pre-registered decision: Tranco legitimate homepages (written before seeing any result)

The URL model flags about a third of real official brand URLs as phishing, because the Kaggle legitimate rows look little like modern official sites. One candidate fix is to add homepages of popular real domains from a pinned Tranco list (list `K9QPW`, generated 2026-09-01, sha256 `611a342b...`) as extra legitimate training rows.

To avoid picking whichever version happens to look best on the test set, the rule is fixed now, before either run is evaluated:

- Train two full runs with identical settings (seed 42): **A** Kaggle only, **B** Kaggle plus the top **20,000** Tranco homepages.
- Compare them on **validation data only** (never the test set, never PhishStats):
  1. false-alarm rate of the selected model on the **validation half** of the official brand URLs, and
  2. PR-AUC on the **Kaggle-origin rows** of the domain-grouped validation split.
- **Adopt B only if** it lowers (1) by at least **5 percentage points** and does not lower (2) by more than **0.01**. Otherwise keep A.
- Whichever wins is then evaluated once with `phishguard evaluate`. Both runs' validation numbers are reported here either way.

### Tranco experiment result (decided by the rule above)

| Run | Selected model | Official-brand validation false-alarm rate | Validation PR-AUC (Kaggle rows) |
|---|---|---|---|
| A: Kaggle only | XGBoost | 37.0% | 0.939 |
| B: + 19,899 Tranco homepages | Random Forest | 40.1% | 0.858 |

B made both criteria worse (false alarms up 3.1 points instead of down 5; PR-AUC down 0.081 instead of within 0.01), so **A is kept** and Tranco stays off by default. Likely reason: in this dataset about half the phishing rows are also bare hostnames, so thousands of "short homepage = legitimate" examples pull real phishing homepages toward legitimate. The option remains available (`phishguard train --extra-legit-tranco N`) for future experiments.

### What was built

- **`phishguard evaluate`** (`evaluation/evaluate.py`, `evaluation/report.py`): one command that reads a finished run and measures everything: held-out metrics for all four models plus a majority-class baseline with 1,000-sample bootstrap 95% CIs, the selected model at the app's own threshold, grouped 5-fold CV, the external sets (official brand test half, PhishStats), the real dashboard's verdicts on those sets, latency, and the test suite with coverage. It writes `metrics/results/evaluation.json` and renders `metrics/VERIFIED_METRICS.md` and `docs/MODEL_CARD.md` from it, so no number is typed by hand.
- **`metrics/reproduce.sh`** is now two commands: `phishguard train --full`, then `phishguard evaluate --with-tests`. Measured: 1,316 seconds on 2 CPU cores.
- **The full run no longer takes hours.** Layer-1 enrichment rewrote its whole checkpoint file every 400 rows; it now checkpoints every 50,000. Full training takes about 12 minutes.
- **Run manifests for full runs.** The pipeline only recorded run mode and augmentation counts for sampled runs; it now always does.
- The pre-rebuild audit moved to `metrics/audit_baseline/` with a README.

### Final numbers (full run, commit `be39bd1`)

All in `metrics/VERIFIED_METRICS.md`. Highlights: XGBoost held-out F1 0.801 (95% CI 0.799 to 0.803), ROC-AUC 0.880, on 147,702 URLs from unseen domains; grouped CV F1 0.828 ± 0.015; 365/365 tests pass.

**Reproducibility check.** The final run was trained twice from scratch (once during development, once through `reproduce.sh` at a clean commit). Every held-out metric matched to the last digit.

### Things the numbers taught us (honest notes)

- **Lower than the 50K check, and that is expected.** The Phase 1 check (50K sample) gave F1 0.823 / ROC-AUC 0.908; the full run gives 0.801 / 0.880. The full test set is 16 times larger, and the 46,712 rows on the 38 evaluation domains (mostly famous sites like google.com and microsoft.com, which are easy legitimate examples) are removed. Grouped CV on 100,000 rows gives ROC-AUC 0.908 ± 0.016, so this single held-out split sits about 1.7 standard deviations below the CV mean (within normal split-to-split variation for grouped data).
- **XGBoost was selected even though LightGBM has a slightly higher test F1** (0.803 vs 0.801). Selection looks only at validation data (XGBoost had the best validation PR-AUC, 0.938). Picking by test F1 would be using the test set to choose, which is the mistake this rebuild removed.
- **Calibration currently hurts slightly.** The isotonic calibrator makes the test Brier score a bit worse (0.1408 raw vs 0.1455 calibrated). Open item: decide on validation data whether to keep it.
- **The URL model alone is weak on famous sites.** Because every evaluation domain is removed from training, the model never sees google.com or amazon.com as legitimate, and Layer 1 flags them. The adjudication layer (which also has the official-domain registry) keeps them out of phishing verdicts. The in-browser demo (Phase 6) must show both layers, not the URL score alone.
- **Without live capture, the dashboard is cautious**: it detects 24.3% of PhishStats URLs as `likely_phishing` and sends the rest to `uncertain`, with 0 false phishing verdicts on official brand URLs. Live capture (Phase 4) is what can move `uncertain` cases either way.

---

## 2026-09-30 · Phase 4: CI and live-capture evaluation

**Goal.** Measure what could not be measured in a sandbox without internet: the full system with real page capture, the legitimacy-rescue layer's effect, full-path latency and the live edge-case suite. And get a CI badge that proves the tests pass on every push.

### What was built

- **`.github/workflows/ci.yml`** runs on every push and pull request:
  - installs the project and runs the full test suite with coverage (golden-output and pipeline-guard tests included);
  - smoke-tests the `phishguard analyze` command with the shipped model;
  - builds the Docker image and runs one analysis inside it (the image could not be built in the earlier sandbox).
- **`phishguard evaluate-live`** (`evaluation/live.py`) captures each page **once** with Playwright, then runs the full analysis **twice** on that capture, with the legitimacy-rescue layer on and off. Same pages, same moment, so the difference is only the rescue layer. URL sets: the 15 live edge cases (now in `data/evaluation/eal_edge_cases.json` with their expected outcomes), the 18 suite URLs, 15 hard-legit URLs, the 136 official brand test-half URLs, the newest PhishStats phishing URLs at run time, and a seeded sample of popular legitimate homepages. It reports pass rates, capture-failure rates (phishing pages are often taken down within hours), the rescue on/off difference and full-path latency, then adds a live-capture section to `metrics/VERIFIED_METRICS.md`.
- **`.github/workflows/live-capture.yml`** runs that evaluation on a GitHub runner. It runs when started by hand from the Actions tab (with options), and once automatically when the evaluator itself changes on this branch; that run commits its results back so they can be reviewed. It is deliberately not on a timer (next point).
- **The verified model now ships in the repo** (`models/layer1/`, 14.6 MB, with `MANIFEST.json`: file hashes, training commit, training-data hash). The app loads `outputs/models/` if you trained your own and otherwise uses the shipped model, so a fresh clone, CI and the Docker image all run the evaluated model. The Random Forest file was re-saved compressed to fit GitHub's size limits (same fitted object).

### Found along the way

- **Capture types a dummy login.** On any page with a username and password field, capture fills `test.user@example.com` / a fake password and submits once, to see where credentials go. That is useful on phishing pages but it also sends failed logins to real sites (LinkedIn, PayPal, bank logins in the evaluation sets). Kept as the default (it is how the system was designed and measured), but there is now a switch, `PHISH_ENABLE_LOGIN_INTERACTION=false`, and the live workflow has a matching option. This is also why the live evaluation does not run on a schedule.
- **Capture folder name.** The Phase 2 import rewrite turned the capture output folder `captures/app_v1` into `captures/phishguard.app`. Now `captures/app`.
- **A failed capture can push toward phishing.** In a smoke test where the browser could not start, even `example.com` came out `likely_phishing`, because the system treats "could not see the page" plus a high URL score as suspicious. The live run reports pass rates both overall and for successfully captured pages only, so this effect is visible rather than hidden.

### First live run (GitHub Actions run 36749733710, commit `2b90195`)

244 URLs, each captured once and analyzed twice. Full table in `metrics/VERIFIED_METRICS.md`; raw rows in `metrics/results/live_capture.json`.

| What | Result |
|---|---|
| Official brand URLs (test half) | 134 of 136 captured; 102 `likely_legitimate`, 26 `uncertain`, **8 `likely_phishing` (5.9%)** |
| Live edge cases | 13 of 15 pass |
| Curated suites | 17 of 18 pass |
| Full-path latency (capture + analysis) | p50 7.1 s, p95 16.1 s |
| Legitimacy rescue on vs off | fired on 86 URLs, **changed 0 final verdicts** |
| Fresh phishing URLs | **none tested**: the PhishStats API returned nothing from the GitHub runner |

What it taught us (each is a real finding, not a tuning target yet):

1. **The rescue layer adjusts scores but never changes a verdict.** It fired on 86 of 244 pages and the final verdict was identical every time, because the Evidence Adjudication Layer decides afterwards. So "the rescue layer reduces false positives" cannot be claimed. It is a candidate for simplification.
2. **Legitimate login flows trip two hard blockers.** Of 14 false phishing verdicts on successfully captured legitimate pages, 9 came from `wrapper_or_interstitial_redirect_pattern` (Wells Fargo, Chase, Outlook, OneDrive sign-in redirects, plus news sites with consent walls) and 4 from `credential_harvesting_pattern` (Dropbox login, AWS console, Stripe dashboard login). These rules treat normal single-sign-on redirects and login forms as abuse.
3. **A failed capture almost always becomes `likely_phishing`.** 19 of 20 legitimate URLs whose capture failed were labeled phishing, mostly infrastructure domains from the Tranco sample that have no website (DNS does not resolve) plus two sites that block headless browsers. "Could not see the page" plus a high URL score is treated as suspicious. That is defensible for unknown domains, but the label claims more certainty than the evidence gives.
4. **Two Weebly phishing edge cases now come back `uncertain`.** The pages were captured, so either the pages changed since April or the rules drifted; it needs a look.

Fixes made after this run:
- **Fresh phishing URLs now have a fallback.** If PhishStats returns nothing, the evaluator uses the OpenPhish public feed and records which source was used.
- **The popular-homepage sample skips domains that do not resolve** (infrastructure domains with no website) and reports how many were skipped, so the set measures real websites.
- Pushing these changes re-runs the live workflow automatically.

Not fixed on purpose: findings 1 to 3 are changes to the adjudication rules. Tuning them while looking at the official-brand **test** half would be tuning on the test set. The honest way is to tune on the **validation** half and confirm once on the test half; that is proposed as the next step.

### Second live run (GitHub Actions run 36758082929, commit `0014d21`)

304 URLs. This time the phishing feed worked (PhishStats answered; the OpenPhish fallback was not needed) and 17 popular-list domains with no website were skipped.

| What | Run 1 | Run 2 |
|---|---|---|
| Official brand URLs (test half), `likely_phishing` | 8 of 136 | 8 of 136 (same 8 pages) |
| All legitimate URLs, `likely_phishing` | 33 of 233 | 25 of 233 |
| of which: capture failed | 19 | 8 |
| of which: wrapper/interstitial blocker | 9 | 10 |
| of which: credential-harvesting blocker | 4 | 4 |
| Popular homepages (Tranco) pass rate | 63.3% | 76.7% (86.8% when the page loaded) |
| Fresh phishing URLs called `likely_phishing` | not tested | **22 of 60 (36.7%)** |
| Fresh phishing URLs called `likely_legitimate` | not tested | **25 of 60** |
| Rescue layer changed a verdict | 0 of 244 | 0 of 304 |
| Full-path latency p50 / p95 | 7.1 s / 16.1 s | 6.8 s / 15.0 s |
| Live edge cases | 13 of 15 | 13 of 15 (same two Weebly pages `uncertain`) |

The new and most important finding is **recall on fresh phishing is low**. The URL model alone put 44 of the 60 above 0.5, but the full system called only 22 phishing and called 25 legitimate. So on this sample the page-level rules lower recall rather than raise it. Looking at the 25:

- 7 had a URL score of 0.8 or higher and were still called legitimate (for example pages on godaddysites.com, square.site, ukit.me, edgeone.dev). These are site-builder pages where the capture did not show a login form, and the rules read "clean page on a known platform" as safe.
- About 10 are the same kit (`/?email=a@a.com&uid=...` and `/point/download/` on random domains). They almost certainly show a harmless page to a data-center browser (cloaking), so the capture never sees the phishing content.
- A few feed entries are not phishing pages at all by the time of capture: real login or captcha pages the phish redirected to (yandex.ru captcha, id.superhuman.com, login.account.rakuten.com, att.com). Feed labels are "reported as phishing", not ground truth.

What this means for claims: the full system's value today is **low false alarms on real brand sites** (8 of 136 flagged), not catching fresh phishing. "Catches X% of live phishing" cannot be claimed. Any tuning must be judged on both sides at once: fewer false alarms on legitimate login pages *and* no further loss of fresh-phishing recall.
## 2026-09-30 · Phase 4b: rule tuning, step 1 (plan and frozen data)

- **The tuning rules were written down before any tuning**: `docs/rebuild/TUNING_PLAN.md` fixes the data, the label hygiene, what code may change (rules only; no allowlisting evaluation sites, no model changes), the keep/reject rule, and that the test split is scored once.
- **Frozen snapshot tooling.** `phishguard snapshot-live` (workflow `live-snapshot.yml`) captures every evaluation URL once and commits the capture records plus page HTML as one gzipped file. `phishguard replay --split val|test|all` runs the full analysis on those saved captures with no network, so before/after numbers compare the same pages. Every test-split replay is logged to `metrics/results/test_access_log.jsonl`.
- **One amendment made before any capture**: a dry run of URL selection showed that grouping fresh phishing by registered domain would collapse every godaddysites.com / sharepoint.com page into one, so grouping and the "popular site" exclusion use the host instead. Recorded in the plan.
- Tests: 370 pass (new `tests/test_snapshot.py` freezes two fake captures and checks two replays give identical verdicts).
