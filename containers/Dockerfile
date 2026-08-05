# syntax=docker/dockerfile:1.7

# Supported upstream pairs are:
#   NGC PyTorch 26.06 (CUDA 13) + cu130-train
#   NGC PyTorch 25.06 (CUDA 12.8) + cu128-train
# Always pass the base image by immutable registry digest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GENET_CODE_REVISION
ARG COSMOS_REVISION=a904d2d36b774a51dd06ff9ff906816b1a04f579
ARG COSMOS_DEPENDENCY_GROUP=cu130-train
ARG UV_VERSION=0.11.28
# Streaming 0.13.0 needs NumPy <2.2 while the Cosmos lock uses 2.2.6.
# The clone is created in the same layer as uv sync to avoid OverlayFS copy-up.
ARG ROBOTWIN_VENV=/opt/genet-preprocess-venv

LABEL org.opencontainers.image.title="GenET Cosmos3-Edge training"
LABEL org.opencontainers.image.revision="${GENET_CODE_REVISION}"
LABEL ai.genet.cosmos.revision="${COSMOS_REVISION}"
LABEL ai.genet.cosmos.dependency-group="${COSMOS_DEPENDENCY_GROUP}"

ENV GENET_CODE_REVISION="${GENET_CODE_REVISION}" \
    GENET_COSMOS_REVISION="${COSMOS_REVISION}" \
    GENET_COSMOS_DEPENDENCY_GROUP="${COSMOS_DEPENDENCY_GROUP}" \
    GENET_BUILD_REVISION_FILE=/opt/genet/GENET_BUILD_REVISION \
    COSMOS_DIR=/opt/cosmos-framework \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libx11-dev \
        tree \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN --mount=type=tmpfs,target=/tmp \
    test -n "${GENET_CODE_REVISION}" \
    && case "${COSMOS_DEPENDENCY_GROUP}" in \
        cu130-train|cu128-train) ;; \
        *) echo "Unsupported COSMOS_DEPENDENCY_GROUP: ${COSMOS_DEPENDENCY_GROUP}" >&2; exit 2 ;; \
       esac \
    && git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cosmos-framework.git \
        "${COSMOS_DIR}" \
    && git -C "${COSMOS_DIR}" fetch --depth 1 origin "${COSMOS_REVISION}" \
    && git -C "${COSMOS_DIR}" checkout --detach "${COSMOS_REVISION}" \
    && test "$(git -C "${COSMOS_DIR}" rev-parse HEAD)" = "${COSMOS_REVISION}" \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        --output /tmp/install-uv.sh \
    && env UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/install-uv.sh \
    && uv_version_output="$(uv --version)" \
    && case "${uv_version_output}" in \
        "uv ${UV_VERSION}"|"uv ${UV_VERSION} "*) ;; \
        *) echo "Unexpected uv version: ${uv_version_output}" >&2; exit 2 ;; \
       esac

WORKDIR ${COSMOS_DIR}
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=tmpfs,target=/tmp \
    uv sync --frozen --all-extras --group="${COSMOS_DEPENDENCY_GROUP}" \
    && .venv/bin/python -c \
        'import numpy, os, torch; group=os.environ["GENET_COSMOS_DEPENDENCY_GROUP"]; expected="13." if group == "cu130-train" else "12.8"; actual=torch.version.cuda or ""; assert actual.startswith(expected), f"{group} resolved CUDA {actual}, expected {expected}"; assert numpy.__version__ == "2.2.6", numpy.__version__' \
    && uv pip freeze --python .venv/bin/python \
        | LC_ALL=C sort > /tmp/genet-training-before.txt \
    && cp -al .venv "${ROBOTWIN_VENV}" \
    && uv pip install \
        --python "${ROBOTWIN_VENV}/bin/python" \
        'mosaicml-streaming==0.13.0' \
    && uv pip check --python "${ROBOTWIN_VENV}/bin/python" \
    && "${ROBOTWIN_VENV}/bin/python" -c \
        'import importlib.metadata as m; import numpy, streaming, torch; assert m.version("mosaicml-streaming") == "0.13.0"; assert tuple(map(int, numpy.__version__.split(".")[:2])) < (2, 2); print(torch.__version__, numpy.__version__)' \
    && uv pip freeze --python .venv/bin/python \
        | LC_ALL=C sort > /tmp/genet-training-after.txt \
    && cmp /tmp/genet-training-before.txt /tmp/genet-training-after.txt

ENV VIRTUAL_ENV=/opt/cosmos-framework/.venv \
    PATH="/opt/cosmos-framework/.venv/bin:${PATH}" \
    LD_LIBRARY_PATH=""

COPY . /opt/genet
WORKDIR /opt/genet
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=tmpfs,target=/tmp \
    test "$(cat "${GENET_BUILD_REVISION_FILE}")" = "${GENET_CODE_REVISION}" \
    && uv pip install \
        --python "${VIRTUAL_ENV}/bin/python" \
        --no-build-isolation \
        --no-deps \
        . \
    && uv pip install \
        --python "${ROBOTWIN_VENV}/bin/python" \
        --no-build-isolation \
        --no-deps \
        . \
    && uv pip check --python "${ROBOTWIN_VENV}/bin/python" \
    && uv pip freeze --python "${VIRTUAL_ENV}/bin/python" \
        | LC_ALL=C sort > /opt/GENET_PIP_FREEZE.txt \
    && sha256sum /opt/GENET_PIP_FREEZE.txt > /opt/GENET_PIP_FREEZE.sha256 \
    && uv pip freeze --python "${ROBOTWIN_VENV}/bin/python" \
        | LC_ALL=C sort > /opt/GENET_PREPROCESS_PIP_FREEZE.txt \
    && sha256sum /opt/GENET_PREPROCESS_PIP_FREEZE.txt \
        > /opt/GENET_PREPROCESS_PIP_FREEZE.sha256 \
    && python -c 'import av, numpy, PIL, torch, cosmos_framework, genet; assert numpy.__version__ == "2.2.6"; print(torch.__version__, torch.version.cuda)' \
    && "${ROBOTWIN_VENV}/bin/python" -c \
        'import importlib.metadata as m; import av, genet, numpy, PIL, streaming, torch; assert m.version("mosaicml-streaming") == "0.13.0"; assert tuple(map(int, numpy.__version__.split(".")[:2])) < (2, 2); print(torch.__version__, numpy.__version__, PIL.__version__)'
