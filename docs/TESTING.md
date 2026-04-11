# Testing strategy

## 1. Pipeline validation

**Goal:** Ingest → clean → enrich → split produce consistent schemas without crashing.

- **Automated:** `pytest tests/ -q` (imports, label mapping, Layer-1 feature row shape).
- **Manual smoke:** After placing a small Kaggle CSV under `data/raw/kaggle/`:

  ```bash
  python -m src.pipeline.run_kaggle_pipeline --limit 500
  ```

- **Checks:** `outputs/reports/kaggle_label_audit.json` (both classes present), `outputs/reports/leakage_audit.json` (zero URL/domain overlap between train and test when using `split_leak_safe`).

## 2. ML-only inference (Layer-1)

**Goal:** `layer1_primary.joblib` loads and returns a probability for an arbitrary URL.

- **Automated:** `tests/test_ml_inference_smoke.py` (skipped if no model on disk).
- **Manual:**

  ```bash
  cd src && python -m app_v1.analyze_dashboard --url "https://example.com" --no-reinforcement
  ```

## 3. Reinforcement layer

**Goal:** Live fetch + org-style signals run in container without crashing on dead hosts.

- **Manual (Docker recommended):** `docker compose up` → dashboard → analyze with reinforcement enabled.
- **CLI:** Same as above **without** `--no-reinforcement` (uses Playwright/HTTP inside the container image).

Expect graceful handling: blocked capture, timeouts, and empty HTML should surface under `evidence_gaps` in the JSON, not stack traces in the UI.

## 4. End-to-end dashboard

**Goal:** Streamlit subprocess path resolves and renders verdict + ML + reinforcement sections.

- Run `streamlit run src/app_v1/frontend.py`, submit a URL, confirm JSON in `outputs/analysis/last_dashboard_analysis.json`.

## 5. Challenge-set evaluation (optional)

**Goal:** Measure Layer-1 (or full enrich) on copied challenge lists—not primary accuracy.

- Copy `archive/sample_data/challenge/*.txt` → `data/raw/`, then `python -m src.pipeline.ingest_challenge` and continue with clean / enrich / optional batch scoring (document any ad-hoc notebook or script you add).

## CI suggestion

On PRs: run `pytest tests/` only. Full pipeline + Playwright jobs stay optional/nightly due to data size and network.
