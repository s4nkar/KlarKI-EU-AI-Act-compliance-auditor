# GPU Setup for KlarKI on Windows (RTX 3050 Ti)

## Prerequisites for Ollama GPU acceleration in Docker

Docker Desktop on Windows requires WSL2 backend + NVIDIA Container Toolkit
to pass the GPU through to containers. Without this, Ollama falls back to
CPU-only inference (~10x slower but still works).

---

## Step 1 — Verify WSL2 is the Docker backend

Open Docker Desktop → Settings → General → confirm "Use WSL 2 based engine" is checked.

---

## Step 2 — Install NVIDIA Container Toolkit in WSL2

Open a WSL2 terminal (`wsl` in PowerShell) and run:

```bash
# Add NVIDIA repo
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

---

## Step 3 — Verify GPU passthrough works

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

You should see your RTX 3050 Ti listed.

---

## Step 4 — Start KlarKI with GPU

```bash
docker compose up -d
```

The `klarki-ollama` service will now use the GPU with `OLLAMA_GPU_LAYERS=20`.

---

## If GPU setup is too complex — CPU fallback

Remove the `deploy.resources` block from `docker-compose.yml` and Ollama
will use CPU. phi3:mini on CPU takes ~15-30 seconds per inference instead
of ~2-5 seconds. For research/demo purposes this is acceptable.

Alternatively, run Ollama **outside Docker** directly on Windows:

```powershell
# Install Ollama from https://ollama.com/download
ollama pull phi3:mini

# Then change OLLAMA_HOST in .env to point to host machine from containers:
# OLLAMA_HOST=http://host.docker.internal:11434
```

This is actually the **easiest option on Windows** — Ollama's Windows build
handles CUDA natively without any Docker GPU configuration.

---

## RAM + VRAM budget at full load

| Service | RAM | VRAM |
|---------|-----|------|
| Docker Desktop + WSL2 | ~1.5 GB | — |
| klarki-api (FastAPI) | ~400 MB | — |
| klarki-chromadb | ~300 MB | — |
| sentence-transformers e5-small | ~350 MB | — |
| Ollama + phi3:mini (20 GPU layers) | ~1.5 GB | ~1.8 GB |
| klarki-frontend (nginx) | ~50 MB | — |
| **Total** | **~4.1 GB** | **~1.8 GB** |

With 15.4 GB total RAM this is fine. VRAM stays under 4 GB with `OLLAMA_GPU_LAYERS=20`.
