# In-browser demo (rebuild Phase 6)

The "Try it" section on [bharathnaveen.com](https://bharathnaveen.com/projects/phishing-detection-system.html#demo) runs the
Layer-1 URL model and the ML-only verdict in the visitor's browser. Nothing is sent to a server, and the URL is never fetched.

## What runs in the browser

| Part | Python it mirrors | How it is exported |
|---|---|---|
| URL canonicalization and the 54 model features | `urls/`, `features/`, CPython `urllib.parse`, `ipaddress`, tldextract 5.3.2 | ported line by line to `js/phishguard-engine.js`; Python's Unicode tables and the ICANN public suffix rules are shipped in `model-core.json` |
| Primary model: XGBoost + isotonic calibrator | `models/layer1/layer1_bundle.joblib` | trees with their float32 split values; calibrator points |
| 4-model agreement: Logistic Regression, Random Forest, XGBoost, LightGBM | `models/layer1/*.joblib` | coefficients; LightGBM trees; Random Forest packed into `model-rf.bin.gz` (thresholds rounded down to float32, which keeps every comparison identical for float32 inputs; leaf values exact) |
| Host identity check | `app/host_path_reasoning.py` | ported |
| Verdict | `app/eal.py` via `app/dashboard.py` with `reinforcement=False` | ported for the ML-only case: with no page capture every page-level input is constant, so only the URL-dependent terms remain. The parity test runs the real dashboard, not this reduction, as the reference. |

"Top contributing signals" are XGBoost's per-feature path contributions (`pred_contribs` with `approx_contribs=True`) recomputed in JS.

## Files

- `export_model.py` writes `build/model-core.json` and `build/model-rf.bin.gz` from the shipped model (hash-checked against `models/layer1/MANIFEST.json`).
- `build_gallery.py` writes `build/gallery.json`: the replay cases and honesty-panel numbers, taken from `metrics/results/tuning/test_after.json`, `test_before.json`, the frozen snapshot and `metrics/results/evaluation.json`.
- `render_portfolio.py` writes `build/demo-section.html` from `gallery.json` and the parity summary.
- `install_portfolio.py <portfolio repo>` copies the assets and inserts the section into the project page and a "Try it" link on the homepage card.
- `parity/`: `build_urls.py` (URL set), `make_reference.py` (Python reference, including the real ML-only dashboard verdict), `run_parity.js` (compares), `results/` (the full run), `fixture_urls.jsonl` (the smaller set CI uses through `tests/test_demo_js_parity.py`).

## Rebuild everything

```bash
python demo/export_model.py
python demo/parity/build_urls.py --out demo/parity/out/urls.jsonl        # needs data/processed/kaggle_test.csv from `phishguard train --full`
# features and model scores for every URL (minutes)
python demo/parity/make_reference.py --urls demo/parity/out/urls.jsonl --out demo/parity/out/full.jsonl
node demo/parity/run_parity.js demo/parity/out/urls.jsonl demo/parity/out/full.jsonl demo/parity/results/parity_full_features_models.json
# the real ML-only dashboard verdict, on 20,000 seeded held-out URLs plus every curated URL (about 20 minutes on 2 CPUs, 2 shards)
python demo/parity/make_reference.py --urls demo/parity/out/urls_dashboard.jsonl --out demo/parity/out/dash.jsonl --dashboard
node demo/parity/run_parity.js demo/parity/out/urls.jsonl demo/parity/out/dash.jsonl demo/parity/results/parity_dashboard_verdicts.json
python demo/build_gallery.py
python demo/render_portfolio.py
python demo/install_portfolio.py ../bharathnaveen-portfolio
```

`build_urls.py` also writes `urls_dashboard.jsonl`: 20,000 held-out rows chosen with `random.Random(42)`, plus every other row.

## Limits worth knowing

- tldextract downloads the public suffix list at run time; the export pins the list it loaded (hash in `model-core.json`). A later list can change the registered domain of a few hosts in Python but not in the demo.
- The demo is the URL model only. The full system also loads the page; the replay gallery shows that part from saved captures.
