#!/usr/bin/env bash
# Start the shared QuickVina HTTP docking service (optional; for parallel / hparam runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

HOST="${ORACLE_HOST:-127.0.0.1}"
PORT="${ORACLE_PORT:-5050}"
ENV_NAME="${ORACLE_CONDA_ENV:-cheml}"
MAX_CONCURRENT_DOCKING="${MAX_CONCURRENT_DOCKING:-160}"
NUM_CPU_DOCK="${NUM_CPU_DOCK:-1}"

echo "Starting benchmark docking oracle on ${HOST}:${PORT} (conda env: ${ENV_NAME})"
echo "MAX_CONCURRENT_DOCKING=${MAX_CONCURRENT_DOCKING} NUM_CPU_DOCK=${NUM_CPU_DOCK}"
echo "Workers: export DOCKING_VINA_URL=http://${HOST}:${PORT}"

export MAX_CONCURRENT_DOCKING NUM_CPU_DOCK
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  exec python "$SCRIPT_DIR/oracle_app.py" --host "$HOST" --port "$PORT"
fi

exec conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/oracle_app.py" \
  --host "$HOST" \
  --port "$PORT"
