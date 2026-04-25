FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Cross-platform PyTorch: ARM64 (Apple Silicon) has no +cpu variant on the WHL index.
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir torch==2.3.1+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.3.1; \
    fi

COPY training/requirements-training.txt .
RUN pip install --no-cache-dir -r requirements-training.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
