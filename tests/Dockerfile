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

# USE_GPU=1 → CUDA 12.1 wheel (faster model inference in tests on GPU machines).
# USE_GPU=0 → CPU-only wheel (default, works everywhere).
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

# Build context is the project root so we can reach api/ requirements.
# Production deps first, then test deps — resolved together to avoid conflicts.
COPY api/requirements.txt api/requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Create the upload directory the FastAPI app writes to during audit integration tests.
# Matches UPLOAD_DIR set in the klarki-test docker-compose service.
RUN mkdir -p /tmp/test-uploads

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
