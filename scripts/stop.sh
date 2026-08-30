#!/usr/bin/env bash
set -euo pipefail

FLASHML_HOME="${FLASHML_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"

if supervisorctl -c "${FLASHML_HOME}/conf/supervisord.conf" shutdown >/dev/null 2>&1; then
  echo "Supervisor shut down."
else
  echo "Supervisor not running or already stopped."
fi
