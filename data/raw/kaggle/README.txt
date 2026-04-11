Primary ML training data (Kaggle)
=================================

See also: docs/DATASET_SETUP.md

Place the CSV from:
  harisudhan411/phishing-and-legitimate-urls
into this folder (any *.csv name is fine).

Alternatively, from the project root with kagglehub installed and Kaggle credentials configured:

  python -m src.pipeline.kaggle_ingest --download

Then run the Kaggle ML pipeline (see README/ARCHITECTURE.md):

  python -m src.pipeline.run_kaggle_pipeline --limit 5000

Label convention on Kaggle (verified in outputs/reports/kaggle_label_audit.json):
  1 = legitimate, 0 = phishing

Internal pipeline labels after ingest:
  0 = legitimate, 1 = phishing
