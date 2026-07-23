# TORCH_IMAGE selects the shared torch layer (docker/torch-base.Dockerfile):
# klarki-torch:gpu (CUDA 12.1 wheel) or klarki-torch:cpu (default). Set
# automatically by run.sh when nvidia-smi is detected; override via
#   docker compose build --build-arg TORCH_IMAGE=klarki-torch:gpu
ARG TORCH_IMAGE=klarki-torch:cpu
FROM ${TORCH_IMAGE} AS torch

FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=torch /wheels /wheels

# Build context is the project root so we can reach api/ requirements.
# torch, prod deps, and test deps are all resolved together in one pip call — see
# docker/torch-base.Dockerfile for why a separate torch install first would
# silently drag in several GB of unrelated CUDA packages.
COPY api/requirements.txt api/requirements-dev.txt ./
RUN pip install --no-cache-dir --find-links=/wheels \
        "$(cat /wheels/torch_spec.txt)" -r requirements.txt -r requirements-dev.txt

# Create the upload directory the FastAPI app writes to during audit integration tests.
# Matches UPLOAD_DIR set in the klarki-test docker-compose service.
RUN mkdir -p /tmp/test-uploads

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
