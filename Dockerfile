FROM python:3.11-slim

# Install system dependencies (build tools and common libs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY pyproject.toml* uv.lock* requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    (pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir .)

# Copy source
COPY . .

# Cloud Run sets PORT; default to 8080 locally
ENV PORT=8080

# Expose for clarity (Cloud Run does not require EXPOSE)
EXPOSE 8080

# Start the bargaining green agent server (PORT comes from env, default 8080)
CMD ["/bin/sh", "-c", "python -m scenarios.bargaining.bargaining_green serve --host 0.0.0.0 --port ${PORT:-8080} --card-url ${CARD_URL:-http://localhost:${PORT:-8080}/}"]

