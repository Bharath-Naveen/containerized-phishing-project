# Archived: AI adjudication path (not used at runtime)

Moved out of `src/` in rebuild Phase 2. Runtime scoring has been deterministic since the Evidence
Adjudication Layer replaced this path: the LLM layer was too costly to run, and its own diagnosis
report found zero verdict changes across 18 test rows.

| File | What it was |
|---|---|
| `src_app_v1/ai_adjudicator.py`, `ai_brand_task.py` | OpenAI calls that nudged the verdict and guessed brand/task |
| `src_app_v1/orchestrator.py`, `compare.py`, `eval_batch.py`, `enrich_dataset.py`, `feature_extract.py`, `url_intel.py`, `verdict.py` | The older screenshot/compare triage flow built around it |
| `src_pipeline/ai_adjudication_audit.py`, `semantic_ai.py` | Audit and helper scripts for the AI layer |
| `tests/` | Its tests, including 3 removed from `tests/test_ml_capture_miss_safety.py` |

Kept for reference and the build story. Imports point at the pre-rebuild layout (`src.app_v1`), so
these files do not run as-is.
