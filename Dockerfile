ARG PYTHON_VERSION=3.12
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-trixie-slim

RUN groupadd --system --gid 1000 nonroot \
 && useradd --system --gid 1000 --uid 1000 --create-home nonroot

WORKDIR /app
RUN chown -R nonroot:0 /app && \
    chgrp -R 0 /app &&  \
    chmod -R g=u /app

# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

ARG PLATFORM_EXTRA=cpu
# RUN --mount=type=cache,target=/root/.cache/uv \
#     --mount=type=bind,source=uv.lock,target=uv.lock \
#     --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
#     uv sync --locked --no-install-project --no-dev

# Installing without the uv.lock temporarily to resolve building for different platforms
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-dev --extra=${PLATFORM_EXTRA}
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked --no-dev --extra=${PLATFORM_EXTRA}


ENV PATH="/app/.venv/bin:$PATH"
USER nonroot
ENTRYPOINT []
COPY . /app

RUN python eval_classifier.py --intents intents.json --testset testset.json --rebuild


CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0",  "--port", "8000", "--proxy-headers"]