# Shared torch wheel cache.
#
# api/Dockerfile, api/Dockerfile.dev, tests/Dockerfile, and training/Dockerfile all
# need torch. This image downloads the wheel (+ its own small deps: numpy, sympy,
# jinja2, etc.) ONCE into /wheels; downstream Dockerfiles COPY --from= that
# directory and pass it to pip via --find-links, so the network fetch happens once
# instead of 3-4 times.
#
# IMPORTANT: downstream installs must pin torch in the SAME `pip install` command
# as the rest of their requirements (see api/Dockerfile etc.) — never install torch
# first and the app's requirements.txt in a second, separate pip call. Splitting it
# into two calls make pip's resolver re-evaluate the transitive `torch>=X` constraint
# from sentence-transformers/etc. against PyPI's *current* torch metadata to build
# the dependency graph, and a plain `torch` requirement on Linux now declares several
# GB of nvidia-cu*/triton CUDA runtime wheels as dependencies — pulling all of that
# in even though the already-installed +cpu build is what actually gets kept. A
# single combined install pins torch as an explicit, exact, top-priority requirement
# from the start, so the resolver never looks at another candidate's metadata.
#
# Build (run.sh does this automatically before building anything that needs it):
#   docker build -t klarki-torch:cpu --build-arg USE_GPU=0 -f docker/torch-base.Dockerfile docker
#   docker build -t klarki-torch:gpu --build-arg USE_GPU=1 -f docker/torch-base.Dockerfile docker
FROM python:3.11-slim

ARG USE_GPU=0
RUN mkdir -p /wheels && arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ] && [ "$USE_GPU" = "1" ]; then \
        spec="torch==2.3.1+cu121"; \
        pip download --no-cache-dir -d /wheels "$spec" \
            --index-url https://download.pytorch.org/whl/cu121; \
    elif [ "$arch" = "x86_64" ]; then \
        spec="torch==2.3.1+cpu"; \
        pip download --no-cache-dir -d /wheels "$spec" \
            --index-url https://download.pytorch.org/whl/cpu; \
    else \
        spec="torch==2.3.1"; \
        pip download --no-cache-dir -d /wheels "$spec"; \
    fi && \
    echo "$spec" > /wheels/torch_spec.txt
