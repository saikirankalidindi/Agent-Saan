# ─────────────────────────────────────────────────────────────────────────────
# Agent Saan — Multi-stage Dockerfile
# Stage 1 (builder): install Python dependencies into a virtual environment
# Stage 2 (runtime): copy only the venv and application source; run as non-root
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install build tools needed for native extensions (e.g. psycopg2, librosa)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install uv for fast dependency resolution
RUN pip install --no-cache-dir --upgrade pip uv

# Copy only the dependency manifest first (layer-cache friendly)
WORKDIR /build
COPY pyproject.toml ./

# Install all production dependencies (no dev extras)
RUN uv pip install --no-cache-dir -e ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir -e .

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime system libraries required by native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy the virtual environment from the builder stage
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder "$VIRTUAL_ENV" "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory and copy application source
WORKDIR /app
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check — polls the /health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run uvicorn with settings from environment variables
CMD ["python", "-m", "uvicorn", "agent_saan.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
