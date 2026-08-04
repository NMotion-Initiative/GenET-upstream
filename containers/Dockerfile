# syntax=docker/dockerfile:1.7

# Pass the NVIDIA Cosmos/PyTorch training image by immutable digest, for example:
#   --build-arg BASE_IMAGE=registry.example/cosmos-train@sha256:<64-hex-digest>
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG GENET_CODE_REVISION
ARG COSMOS_REVISION=a904d2d36b774a51dd06ff9ff906816b1a04f579

LABEL org.opencontainers.image.title="GenET Cosmos3-Edge training"
LABEL org.opencontainers.image.revision="${GENET_CODE_REVISION}"
LABEL ai.genet.cosmos.revision="${COSMOS_REVISION}"

ENV GENET_CODE_REVISION="${GENET_CODE_REVISION}" \
    GENET_COSMOS_REVISION="${COSMOS_REVISION}" \
    GENET_BUILD_REVISION_FILE=/opt/genet/GENET_BUILD_REVISION \
    COSMOS_DIR=/opt/cosmos-framework \
    PYTHONUNBUFFERED=1

WORKDIR /opt
RUN test -n "${GENET_CODE_REVISION}" \
    && git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cosmos-framework.git \
        "${COSMOS_DIR}" \
    && git -C "${COSMOS_DIR}" fetch --depth 1 origin "${COSMOS_REVISION}" \
    && git -C "${COSMOS_DIR}" checkout --detach "${COSMOS_REVISION}" \
    && test "$(git -C "${COSMOS_DIR}" rev-parse HEAD)" = "${COSMOS_REVISION}" \
    && python -m pip install "${COSMOS_DIR}[train]"

COPY . /opt/genet
WORKDIR /opt/genet
RUN test "$(cat "${GENET_BUILD_REVISION_FILE}")" = "${GENET_CODE_REVISION}" \
    && python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/genet-wheel . \
    && python -m pip install /tmp/genet-wheel/genet-*.whl \
    && python -m pip freeze --all | LC_ALL=C sort > /opt/GENET_PIP_FREEZE.txt \
    && sha256sum /opt/GENET_PIP_FREEZE.txt > /opt/GENET_PIP_FREEZE.sha256 \
    && rm -rf /tmp/genet-wheel

WORKDIR /opt/genet
