# Multi-stage build for better caching
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies (this layer gets cached)
FROM base as system-deps
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python dependencies layer (gets cached unless requirements.txt changes)
FROM system-deps as python-deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final application layer
FROM python-deps as app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/storage/historical /app/storage/models \
    /app/storage/backups /app/storage/exports

# Copy application code (this layer changes most frequently)
COPY . .

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading && \
    chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

EXPOSE 8000
CMD ["python", "main.py"]