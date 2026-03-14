# Use official pure Python 3.12 slim image
FROM python:3.12-slim-bookworm AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies required by tokenizers/asyncpg/river C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for builder
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies (only production, no test/train packages)
RUN pip wheel --no-deps --requirement <(python -c "import tomli; print('\n'.join(tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" 2>/dev/null || pip install tomli && python -c "import tomli; print('\n'.join(tomli.load(open('pyproject.toml', 'rb'))['project']['dependencies']))") --wheel-dir /app/wheels
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels .

# --- Final Production Stage ---
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies for psycopg/asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -U appuser

WORKDIR /app

# Copy pre-compiled wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache --no-index /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser config.yaml config.yaml
COPY --chown=appuser:appuser config-example.yaml config-example.yaml

# Switch to non-root user
USER appuser

CMD ["uvicorn", "sentistream.ingestion.api:app", "--host", "0.0.0.0", "--port", "8000"]
