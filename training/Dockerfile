# TORCH_IMAGE selects the shared torch layer (docker/torch-base.Dockerfile):
# klarki-torch:gpu (CUDA 12.1 wheel, GPU training) or klarki-torch:cpu (default).
# Set automatically by run.sh when nvidia-smi is detected; override via
#   docker compose build --build-arg TORCH_IMAGE=klarki-torch:gpu
ARG TORCH_IMAGE=klarki-torch:cpu
FROM ${TORCH_IMAGE} AS torch

FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=torch /wheels /wheels

# torch and the rest of requirements-training.txt are resolved together in one pip
# call — see docker/torch-base.Dockerfile for why a separate torch install first
# would silently drag in several GB of unrelated CUDA packages.
COPY requirements-training.txt .
RUN pip install --no-cache-dir --find-links=/wheels \
        "$(cat /wheels/torch_spec.txt)" -r requirements-training.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
