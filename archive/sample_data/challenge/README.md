# Sample challenge / evaluation URL lists

Small curated `legit-*.txt` and `phish-*.txt` files for **optional** brand-impersonation and challenge-set workflows. They are **not** used as the primary Kaggle training distribution.

**Use locally**

1. Copy (do not move, if you want to keep the archive intact) into `data/raw/`:

   ```powershell
   Copy-Item archive\sample_data\challenge\*.txt data\raw\
   ```

   ```bash
   cp archive/sample_data/challenge/*.txt data/raw/
   ```

2. Run challenge ingest and follow `README.md` → “Challenge-set evaluation”.

These files are **gitignored** under `data/raw/` once copied.
