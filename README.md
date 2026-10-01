# Phishing Detection System

[![CI](https://github.com/Bharath-Naveen/phishing-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Bharath-Naveen/phishing-detection-system/actions/workflows/ci.yml)

An explainable phishing detector. Give it a URL and it returns **likely phishing**, **uncertain** or **likely legitimate**, with the reasons behind the verdict. A fast machine-learning model scores the URL, a headless browser loads the page, and a rule-based judge weighs all the evidence. When the evidence disagrees, it says "uncertain" instead of guessing.

**[Try the live demo](https://bharathnaveen.com/projects/phishing-detection-system.html#demo)**: paste any URL and the URL model scores it in your browser. Nothing is sent to a server.

Built solo as an MS capstone, then rebuilt so that every number below can be regenerated with one script.

## Results

| What was measured | Result |
|---|---|
| URL model (XGBoost) on 147,702 test URLs from websites it never saw in training | F1 0.801 (95% CI 0.799 to 0.803), ROC-AUC 0.880, precision 0.772, recall 0.832 |
| 136 real brand sites (PayPal, Microsoft, Chase, Amazon and others), URL model alone | 40.4% flagged as phishing |
| The same 136 sites, full system with page capture | 4 called phishing, 24 uncertain, 108 legitimate |
| 99 fresh phishing pages from the PhishStats feed, full system | 39 called phishing, 20 uncertain, 40 legitimate |
| Time per URL | 15.7 ms for the URL model, 6.8 s median with live page capture |
| Automated tests | 373, all passing, run in GitHub Actions on every push along with a Docker build |

**What it is good at, and what it is not.** The layers do their job of not crying wolf on real sites: the URL model alone would flag 40.4% of real brand pages, and the full system flags 4 of 136. Catching brand-new phishing is the weak spot (39 of 99). The URL model scores many famous homepages as high as phishing pages, so no rule can separate them without adding false alarms. A better URL model is the next step.

Full tables, with the command, commit, data hash and environment that produced them: [metrics/VERIFIED_METRICS.md](metrics/VERIFIED_METRICS.md). Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## How it works

A URL passes through five layers:

1. **URL model and model agreement.** XGBoost scores 54 features of the URL text (lengths, entropy, lure words, brand names in the wrong place, free hosting). Three more models (Logistic Regression, Random Forest, LightGBM) vote as supporting evidence. No network access is needed.
2. **Live capture.** Playwright loads the page and records the final URL, redirects, form targets and TLS state.
3. **HTML and behavior.** Login harvesters, wrapper pages, cross-domain forms and suspicious scripts are detected from the captured page.
4. **Brand and trust context.** Does the brand on the page match the domain? A small registry of official domains is used as a weak prior, never as a whitelist.
5. **Evidence Adjudication Layer.** A deterministic judge adds up phishing, legitimacy and ambiguity signals, applies hard blockers (for example a password form posting to another domain) and returns the verdict with its reasons.

No external AI service is called at runtime, so the same input always gives the same verdict.

More detail in plain language: [docs/rebuild/HOW_IT_WORKS.md](docs/rebuild/HOW_IT_WORKS.md).

## Why the numbers can be trusted

An audit of the first version found a leaky train/test split and other problems, so training and evaluation were rebuilt. The full record is in [docs/rebuild/REBUILD_LOG.md](docs/rebuild/REBUILD_LOG.md).

- **Split by website.** Train and test are split by registered domain, so no website is in both. A guard stops the run if any domain overlaps.
- **No dataset shortcut.** In this dataset `https` is far more common on phishing rows, so features are computed without the scheme.
- **Evaluation stays out of training.** Every curated evaluation URL is removed from the training data.
- **Model chosen on validation data.** The test set is never used to pick a model or tune a rule.
- **Rule tuning written down first.** The plan was fixed before any tuning ([docs/rebuild/TUNING_PLAN.md](docs/rebuild/TUNING_PLAN.md)), rules were tuned on a validation half of one saved snapshot of real pages, and the test half was scored once.
- **One command reproduces everything.** `bash metrics/reproduce.sh` retrains and re-evaluates from scratch (seed 42, about 25 minutes on 2 CPU cores).
- **The demo is the evaluated model.** The in-browser version is checked against the Python app: identical features and scores on 148,746 URLs and identical verdicts on 21,032 ([demo/](demo/)).

## Run it

With Docker (the image includes the browser and the verified model):

```bash
docker compose up --build
```

Then open http://localhost:8501.

Or locally with Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
playwright install chromium
phishguard serve                 # dashboard on http://localhost:8501
```

The `phishguard` command:

| Command | What it does |
|---|---|
| `phishguard analyze --url "https://example.com"` | Score one URL (add `--no-reinforcement` to skip the page capture) |
| `phishguard serve` | Start the dashboard |
| `phishguard train` | Train on a 50,000-row sample (`--full` for the whole dataset; needs the Kaggle CSV, see [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md)) |
| `phishguard evaluate` | Regenerate the verified metrics and the model card |
| `phishguard evaluate-live` | Evaluate the full system with live page capture (run by GitHub Actions) |
| `pytest` | Run the tests, including golden-output tests that pin the dashboard's behavior |

Live capture fills a dummy login on pages with a password field, to see where credentials go. Set `PHISH_ENABLE_LOGIN_INTERACTION=false` to turn that off.

## Repo layout

```text
src/phishguard/     the package: urls, data, features, models, pipelines, evaluation, app (dashboard, capture, judge)
models/layer1/      the verified model the app ships, with hashes in MANIFEST.json
metrics/            verified metrics, raw results and the reproduce script
data/evaluation/    evaluation URL sets and the frozen snapshot of real pages
demo/               export, parity tests and page builder for the in-browser demo
tests/              unit, regression, golden-output and JS parity tests
docs/               how it works, rebuild log, tuning plan, model card
archive/legacy/     retired code (an earlier AI adjudication path, old scripts)
```

## Limitations

- Recall on fresh phishing is low, as shown above.
- A URL-only model cannot see page content, so it over-flags modern legitimate sites; the page layers correct most of that.
- Live capture can be blocked by bot checks or shown harmless content by phishing kits, and phishing pages are often taken down within hours.
- When a page cannot be loaded, a high URL score usually ends as "likely phishing", which claims more certainty than the evidence gives.
- The training data has no dates, so performance on future campaigns is measured only on the PhishStats samples.
