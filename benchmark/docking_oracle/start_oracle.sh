#!/usr/bin/env bash
# Start the shared QuickVina HTTP docking service (optional; for parallel / hparam runs).
#
# Env:
#   ORACLE_HOST, ORACLE_PORT     bind address (default 127.0.0.1:5050)
#   ORACLE_CONDA_ENV             conda env (default cheml)
#   ORACLE_THREADS               gunicorn gthread workers (default 100)
#   NUM_CPU_DOCK                 CPUs per qvina call (default 1)
#   MAX_CONCURRENT_DOCKING       in-process docking semaphore (default 100)
#   ORACLE_SERVER                gunicorn (default) or flask
#   ORACLE_EXHAUSTIVENESS        optional override passed to DockingOracle
#   DOCKING_TMP_DIR              per-instance temp dir (default $OUT_DIR/vina_oracle_$PORT)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

HOST="${ORACLE_HOST:-127.0.0.1}"
PORT="${ORACLE_PORT:-5050}"
ENV_NAME="${ORACLE_CONDA_ENV:-cheml}"
THREADS="${ORACLE_THREADS:-100}"
SERVER="${ORACLE_SERVER:-gunicorn}"
export NUM_CPU_DOCK="${NUM_CPU_DOCK:-1}"
export MAX_CONCURRENT_DOCKING="${MAX_CONCURRENT_DOCKING:-100}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results}"
export DOCKING_TMP_DIR="${DOCKING_TMP_DIR:-$OUT_DIR/vina_oracle_${PORT}}"
mkdir -p "$DOCKING_TMP_DIR"

echo "Starting benchmark docking oracle on ${HOST}:${PORT}"
echo "  conda env: ${ENV_NAME}"
echo "  server: ${SERVER}"
echo "  NUM_CPU_DOCK=${NUM_CPU_DOCK}"
echo "  MAX_CONCURRENT_DOCKING=${MAX_CONCURRENT_DOCKING}"
echo "  ORACLE_THREADS=${THREADS}"
echo "  DOCKING_TMP_DIR=${DOCKING_TMP_DIR}"
echo "Workers: export DOCKING_VINA_URL=http://${HOST}:${PORT}"

cd "$PROJECT_ROOT"

if [[ "$SERVER" == "flask" ]]; then
  exec conda run --no-capture-output -n "$ENV_NAME" python "$SCRIPT_DIR/oracle_app.py" \
    --host "$HOST" \
    --port "$PORT"
fi

GUNICORN_ARGS=(
  --bind "${HOST}:${PORT}"
  --workers 1
  --threads "$THREADS"
  --worker-class gthread
  --timeout 1800
  --graceful-timeout 30
  --keep-alive 5
  --access-logfile -
  --error-logfile -
  "benchmark.docking_oracle.oracle_app:app"
)

exec conda run --no-capture-output -n "$ENV_NAME" gunicorn "${GUNICORN_ARGS[@]}"
