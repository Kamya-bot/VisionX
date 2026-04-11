FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache bust - forces full rebuild
ARG CACHEBUST=4

# Copy all code
COPY . .

# Create runtime dirs and train models (run from /app where training/ folder is)
RUN mkdir -p data/raw data/processed trained_models logs mlruns
RUN python training/generate_dataset.py && \
    python training/train_models.py || \
    echo "Model training skipped"

# Switch into the app package so imports resolve correctly
WORKDIR /app/app

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:10000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]

