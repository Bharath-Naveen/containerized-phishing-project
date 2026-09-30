# Playwright provides Chromium + system deps; version must match requirements.txt (playwright==…).
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

# Application code (see .dockerignore)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium

COPY src/ ./src/
# Small runtime registries the app needs (large datasets stay out of the image; mount ./data for those).
COPY data/official_domains.json ./data/official_domains.json
COPY data/reference/ ./data/reference/
COPY data/evaluation/ ./data/evaluation/
# The verified model shipped with the repo (see models/layer1/MANIFEST.json).
COPY models/layer1/ ./models/layer1/

# The phishguard package lives in /app/src.
ENV PYTHONPATH=/app/src
ENV PHISH_PROJECT_ROOT=/app
ENV PHISH_DATA_DIR=/app/data
ENV PHISH_OUTPUTS_DIR=/app/outputs
ENV PHISH_LOGS_DIR=/app/logs
# Captures / HTML / screenshots (override in compose with volume)
ENV PHISH_OUTPUT_DIR=/data/captures

EXPOSE 8501

# Default command supports cloud PORT env with local fallback.
CMD ["sh", "-c", "streamlit run src/phishguard/app/frontend.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
