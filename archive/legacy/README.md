# Legacy screenshot-first code

These artifacts are **not** part of the primary Layer-1 ML + dashboard flow. They are kept for reference, demos, or research.

| Item | Description |
|------|-------------|
| `frontend_screenshot_legacy.py` | Streamlit UI that runs `app_v1.orchestrator` (capture → compare → verdict). Run from repo root with `PYTHONPATH` including `src`: `streamlit run archive/legacy/frontend_screenshot_legacy.py` |
| `scripts/*.py` | Old one-off scripts (scrapers, seed builders) superseded by `src/pipeline`. |

Primary app: `streamlit run src/app_v1/frontend.py`.
