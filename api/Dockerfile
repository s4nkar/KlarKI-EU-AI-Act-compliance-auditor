# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Installs all Python dependencies (including compiling C extensions).
# gcc and build tools stay in this stage only — not in the final image.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch wheel, pre-fetched by the shared klarki-torch:cpu image
# (docker/torch-base.Dockerfile) instead of downloading it here — avoids fetching
# the same ~200 MB wheel in every image that needs torch.
# klarki-torch:cpu must exist locally first: ./run.sh builds it automatically.
COPY --from=klarki-torch:cpu /wheels /wheels

# torch is installed in this SAME command as requirements.txt (not a separate RUN) —
# see docker/torch-base.Dockerfile for why splitting it into two pip calls silently
# drags in several GB of unrelated CUDA packages.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install --find-links=/wheels \
        "$(cat /wheels/torch_spec.txt)" -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Minimal final image — no gcc, no build tools, ~40% smaller than single-stage.
FROM python:3.11-slim

WORKDIR /app

# WeasyPrint runtime libs (Pango/Cairo for PDF rendering) — runtime only, no dev headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from builder (torch + deps + requirements)
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

RUN mkdir -p /data/uploads

# Pin HF cache to a fixed path so both the build-time download (root) and the
# runtime user (appuser) resolve the same location.
ENV HF_HOME=/app/.cache/huggingface

# Pre-download models that are loaded at startup so HF_HUB_OFFLINE=1 works at runtime.
# This bakes them into the image layer — no network access needed when the container runs.
RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
