# Playwright provides Chromium + system deps; version must match requirements.txt (playwright==…).
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

# Application code (see .dockerignore)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium

COPY src/ ./src/

# /app for `python -m src.pipeline.*`; /app/src for `python -m app_v1.*`.
ENV PYTHONPATH=/app:/app/src
ENV PHISH_PROJECT_ROOT=/app
ENV PHISH_DATA_DIR=/app/data
ENV PHISH_OUTPUTS_DIR=/app/outputs
ENV PHISH_LOGS_DIR=/app/logs
# Captures / HTML / screenshots (override in compose with volume)
ENV PHISH_OUTPUT_DIR=/data/captures

EXPOSE 8501

# Default command supports cloud PORT env with local fallback.
CMD ["sh", "-c", "streamlit run src/app_v1/frontend.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
