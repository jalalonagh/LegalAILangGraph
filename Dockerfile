# =============================================================================
# Production Dockerfile
# =============================================================================
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        libffi-dev \
        git \
        curl \
        ca-certificates \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-deu \
        tesseract-ocr-fra \
        tesseract-ocr-spa \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r -g 1000 legalai \
    && useradd -r -m -u 1000 -g legalai -s /bin/bash legalai

# Set working directory
WORKDIR /app

# Create a venv
ENV VENV=/opt/venv
RUN python -m venv $VENV --copies --upgrade \
    && $VENV/bin/pip install --no-cache-dir --upgrade pip

ENV PATH="$VENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Copy lockfiles / project files
COPY pyproject.toml ./

# Copy application source
COPY app/ ./app/
COPY migrations/ ./migrations/

RUN pip install --no-cache-dir -e ".[all]"

# Install OCR / Tesseract data
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.0.0

# Create temporary upload directory
RUN mkdir -p /tmp/legalai_uploads \
    && chown -R legalai:legalai /app /tmp/legalai_uploads /opt/venv

# Switch to non-root user
USER legalai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health/live || exit 1

EXPOSE 8000
ENTRYPOINT ["gunicorn"]
CMD ["-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--bind", "0.0.0.0:8000", "app.main:app"]
