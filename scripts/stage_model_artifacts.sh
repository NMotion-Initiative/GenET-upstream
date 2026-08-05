#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly DEFAULT_SPEC="${PROJECT_ROOT}/configs/checkpoints/cosmos3_edge.json"

exec python -m genet.cli.stage_artifacts --spec "${DEFAULT_SPEC}" "$@"
