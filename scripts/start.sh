#!/usr/bin/env bash
set -euo pipefail

FLASHML_HOME="${FLASHML_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
if [ -z "${CONDA_ROOT:-}" ]; then
  if [ -d "/venv" ]; then
    CONDA_ROOT="/venv"
  elif [ -d "/opt/conda/envs" ]; then
    CONDA_ROOT="/opt/conda/envs"
  elif [ -d "${HOME}/miniconda3/envs" ]; then
    CONDA_ROOT="${HOME}/miniconda3/envs"
  else
    CONDA_ROOT="/venv"
  fi
fi
export FLASHML_HOME CONDA_ROOT PYTHONUNBUFFERED=1

if [ ! -f "${FLASHML_HOME}/conf/supervisord.conf" ]; then
  if [ -f "${FLASHML_HOME}/conf/supervisord.conf.example" ]; then
    echo "Creating conf/supervisord.conf from conf/supervisord.conf.example..."
    cp "${FLASHML_HOME}/conf/supervisord.conf.example" "${FLASHML_HOME}/conf/supervisord.conf"
  else
    echo "Error: ${FLASHML_HOME}/conf/supervisord.conf not found." >&2
    exit 1
  fi
fi

export FLASHML_MOGE_ROOT="${FLASHML_MOGE_ROOT:-${FLASHML_HOME}/third_party/MoGe}"
export FLASHML_SIMPLECLICK_ROOT="${FLASHML_SIMPLECLICK_ROOT:-${FLASHML_HOME}/third_party/SimpleClick}"
export FLASHML_SIMPLECLICK_CHECKPOINT="${FLASHML_SIMPLECLICK_CHECKPOINT:-${FLASHML_HOME}/weights/simpleclick/cocolvis_vit_huge.pth}"
export FLASHML_ONEFORMER_MODEL_DIR="${FLASHML_ONEFORMER_MODEL_DIR:-${FLASHML_HOME}/weights/oneformer}"
export FLASHML_LAMA_MODEL_DIR="${FLASHML_LAMA_MODEL_DIR:-${FLASHML_HOME}/weights/lama}"
export HF_HOME="${HF_HOME:-${FLASHML_HOME}/weights/huggingface}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"

mkdir -p "${FLASHML_HOME}/logs"

if ! command -v supervisord >/dev/null 2>&1; then
  echo "supervisord not found. Install supervisor (apt install supervisor) or re-run scripts/setup_conda.sh as root." >&2
  exit 1
fi

if [ ! -x "${CONDA_ROOT}/flashml-api/bin/uvicorn" ]; then
  echo "Conda envs missing at ${CONDA_ROOT}. Run scripts/setup_conda.sh first." >&2
  exit 1
fi

exec supervisord -c "${FLASHML_HOME}/conf/supervisord.conf"
