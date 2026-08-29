#!/usr/bin/env bash
set -euo pipefail

FLASHML_HOME="${FLASHML_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export FLASHML_HOME CONDA_ROOT PYTHONUNBUFFERED=1

if [ -f "${FLASHML_HOME}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${FLASHML_HOME}/.env"
  set +a
fi

export FLASHML_MOGE_ROOT="${FLASHML_MOGE_ROOT:-${FLASHML_HOME}/third_party/MoGe}"
export FLASHML_SIMPLECLICK_ROOT="${FLASHML_SIMPLECLICK_ROOT:-${FLASHML_HOME}/third_party/SimpleClick}"
export FLASHML_SIMPLECLICK_CHECKPOINT="${FLASHML_SIMPLECLICK_CHECKPOINT:-${FLASHML_HOME}/weights/simpleclick/cocolvis_vit_huge.pth}"
export FLASHML_ONEFORMER_MODEL_DIR="${FLASHML_ONEFORMER_MODEL_DIR:-${FLASHML_HOME}/weights/oneformer}"
export HF_HOME="${HF_HOME:-${FLASHML_HOME}/weights/huggingface}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

mkdir -p "${FLASHML_HOME}/logs"

if ! command -v supervisord >/dev/null 2>&1; then
  echo "supervisord not found. Install supervisor (apt install supervisor) or re-run scripts/setup_conda.sh as root." >&2
  exit 1
fi

if [ ! -x "${CONDA_ROOT}/envs/flashml-api/bin/uvicorn" ]; then
  echo "Conda envs missing. Run scripts/setup_conda.sh first." >&2
  exit 1
fi

exec supervisord -c "${FLASHML_HOME}/conf/supervisord.conf"
