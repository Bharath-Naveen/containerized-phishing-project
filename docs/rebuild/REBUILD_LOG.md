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

## 2026-09-30 · Phase 3: Retrain and evaluate (in progress)

### Pre-registered decision: Tranco legitimate homepages (written before seeing any result)

The URL model flags about a third of real official brand URLs as phishing, because the Kaggle legitimate rows look little like modern official sites. One candidate fix is to add homepages of popular real domains from a pinned Tranco list (list `K9QPW`, generated 2026-09-01, sha256 `611a342b...`) as extra legitimate training rows.

To avoid picking whichever version happens to look best on the test set, the rule is fixed now, before either run is evaluated:

- Train two full runs with identical settings (seed 42): **A** Kaggle only, **B** Kaggle plus the top **20,000** Tranco homepages.
- Compare them on **validation data only** (never the test set, never PhishStats):
  1. false-alarm rate of the selected model on the **validation half** of the official brand URLs, and
  2. PR-AUC on the **Kaggle-origin rows** of the domain-grouped validation split.
- **Adopt B only if** it lowers (1) by at least **5 percentage points** and does not lower (2) by more than **0.01**. Otherwise keep A.
- Whichever wins is then evaluated once with `phishguard evaluate`. Both runs' validation numbers are reported here either way.
