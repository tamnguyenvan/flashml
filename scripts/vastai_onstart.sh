#!/usr/bin/env bash
# Vast.ai on-start entrypoint. Copy this into the instance On-Start script,
# or run it after the repo is present on the machine.
set -euo pipefail

export FLASHML_HOME="${FLASHML_HOME:-/workspace/flashml}"
export CONDA_ROOT="${CONDA_ROOT:-/workspace/miniconda3}"

if [ ! -f "${FLASHML_HOME}/scripts/setup_conda.sh" ]; then
  echo "FlashML is not at ${FLASHML_HOME}. Clone the repo there first." >&2
  exit 1
fi

bash "${FLASHML_HOME}/scripts/setup_conda.sh"
exec bash "${FLASHML_HOME}/scripts/start.sh"
