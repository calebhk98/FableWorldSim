# Headless FableWorldSim core image (Linux).
#
# Docker/WSL2 is an explicit OPTION, not the assumed answer: this image is
# the clean home for native bits (DGGRID, CUDA) on any host, while native
# Windows (H3 + CPU, GPU optional) must keep working without containers.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t fableworldsim .
# Run:
#   docker run -p 8080:8080 fableworldsim

FROM python:3.11-slim

WORKDIR /app

# gcc enables the native flow-accumulation kernel (auto-falls back without it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core core
COPY ports ports
COPY adapters adapters
COPY api api
COPY config config
COPY content content

RUN pip install --no-cache-dir ".[api,grid-h3,grid-s2]"

EXPOSE 8080

# --factory: api.app:create_app builds the app (hydrating config/ TOML).
CMD ["uvicorn", "--factory", "api.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
