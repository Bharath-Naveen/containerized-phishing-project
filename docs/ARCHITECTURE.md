# Architecture (post-pivot)

## Goals

1. **Layer-1 ML triage** — fast phishing vs legitimate classifier from **URL + host (+ optional DNS)** features only. **No browser required** for training or inference.
2. **Layer-2 reinforcement** — containerized live fetch (Playwright / HTTP fallback), DNS/redirect visibility, DOM/login heuristics, and **org-style** signals (free hosting, brand/domain mismatch, sensitive copy).
3. **Layer-3 explanation** — dashboard JSON + UI: verdict, scores, contributing signals, and explicit **evidence gaps** when the page did not load.

Screenshot **comparison** against a captured legit reference is **deprecated** for product UX; the old Streamlit entrypoint lives under **`archive/legacy/frontend_screenshot_legacy.py`** (see `archive/legacy/README.md`).

## Data sources

| Source | Role |
|--------|------|
| **Kaggle** `harisudhan411/phishing-and-legitimate-urls` (under `data/raw/kaggle/`) | **Primary** supervised training (binary labels). |
| **`phish-*.txt` / `legit-*.txt`** brand files | **Challenge / eval / demos** only (`python -m src.pipeline.ingest_challenge`). Not merged into the primary Kaggle training path. |

**Label conventions**

- **Kaggle file:** `1` = legitimate, `0` = phishing (verified in `outputs/reports/kaggle_label_audit.json`).
- **Internal pipeline / models:** `0` = legitimate, `1` = phishing (`src/pipeline/label_policy.py`).

## Batch ML path (primary)

```text
kaggle_ingest → clean (dedupe canonical_url) → enrich --layer1-only → split_leak_safe → train --layer1-only → metrics
```

Orchestrator CLI: `python -m src.pipeline.run_kaggle_pipeline [--limit N]`

- **Deduplication:** `clean.py` on `canonical_url`.
- **Leak-aware split:** `StratifiedGroupKFold` on **registered domain** (derived from URL if needed) in `split_leak_safe.py`.
- **Leakage audit:** `outputs/reports/leakage_audit.json`.
- **Training excludes** filename/metadata columns (`source_file`, `source_dataset`, `source_brand_hint`, `action_category`, `kaggle_raw_status`, …) — see `src/pipeline/train.py`.

## Interactive app

- **Dashboard:** `streamlit run src/app_v1/frontend.py` (subprocess `python -m app_v1.analyze_dashboard`).
- **Artifacts:** `outputs/analysis/last_dashboard_analysis.json`.

Live URL work runs **inside Docker** in the recommended deployment (`docker compose up`).

## Reuse vs deprecate

- **Reuse:** Docker/compose, `clean`, feature extractors (`url_features`, `hosting_features`, `dns_features`), `split` (legacy path), `train`, enrichment HTTP/Playwright stack for Layer-2.
- **Deprecated (UX):** side-by-side screenshot comparison as primary story (`archive/legacy/frontend_screenshot_legacy.py`; `compare.py` still used by legacy `orchestrator` JSONL flow).
- **New:** `kaggle_ingest`, `ingest_challenge`, `layer1_features`, `split_leak_safe`, `leakage_report`, `run_kaggle_pipeline`, `ml_layer1`, `org_style_signals`, `analyze_dashboard`, dashboard `frontend.py`.

## Optional

- **Multi-seed eval:** `python -m src.pipeline.eval_multi_seed --input data/processed/enriched_kaggle_layer1.csv --seeds 42,43,44`
- **Kaggle download:** `pip install kagglehub` then `python -m src.pipeline.kaggle_ingest --download`
