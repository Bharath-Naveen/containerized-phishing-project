# Archived: superseded pipeline scripts

Moved out of `src/pipeline/` in rebuild Phase 2. None were reachable from the app, the Kaggle
pipeline, the fresh-data retrain or the evaluation scripts.

| File | Superseded by |
|---|---|
| `ingest.py`, `ingest_challenge.py` | Kaggle ingest (`phishguard.data.kaggle`) |
| `split.py` | Domain-grouped split with guards (`phishguard.data.split`) |
| `run_all.py`, `prepare_ml_dataset.py`, `balance_training.py` | `phishguard train` pipeline |
| `analyze_dataset.py`, `eval_multi_seed.py` | Phase 3 evaluation command |

Imports point at the pre-rebuild layout, so these do not run as-is.
