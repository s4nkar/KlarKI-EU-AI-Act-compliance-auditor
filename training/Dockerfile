FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# USE_GPU=1 → CUDA 12.1 wheel (GPU training, ~2.5 GB image delta).
# USE_GPU=0 → CPU-only wheel (default, smaller image, works everywhere).
# Set automatically by run.sh when nvidia-smi is detected; override via
#   docker compose build --build-arg USE_GPU=1
ARG USE_GPU=0
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ] && [ "$USE_GPU" = "1" ]; then \
        pip install --no-cache-dir torch==2.3.1+cu121 \
            --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir torch==2.3.1+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.3.1; \
    fi

COPY requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
