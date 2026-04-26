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

# CPU-only PyTorch — avoids the 2.6 GB CUDA wheel on x86_64.
# ARM64 (Apple Silicon / linux/aarch64) has no +cpu variant on the WHL index;
# standard PyPI torch is already CPU-only on that platform.
# Use --prefix=/install so torch and all its deps (typing_extensions, numpy…)
# land in the prefix and get copied to Stage 2.
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir --prefix=/install torch==2.3.1+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir --prefix=/install torch==2.3.1; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


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

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
