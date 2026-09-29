#!/usr/bin/env bash
# Launch one ChemLactica lead-opt wave: three seed molecules of one target,
# each with RNG seeds 1-5, each pinned to one Vina oracle.
#
# GPU layout (5 GPUs): all three seed-molecule jobs that share an RNG seed
# land on the same GPU (seed 1 → GPU 0, seed 2 → GPU 1, ..., seed 5 → GPU 4).
#
# Usage:
#   ./genetic_chemalactica/scripts/run_lead_wave.sh [target]
# Defaults: parp1  →  lead.parp1_06_0, lead.parp1_06_1, lead.parp1_06_2
#
# Similarity threshold is 0.6 (task suffix _06_). Oracle budget 10000.
# Starts three Vina services on 5050/5051/5052 if they are not already healthy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results}"

TARGET="${1:-parp1}"
SEEDMOLS=(${LEAD_SEEDMOLS:-0 1 2})
SIM="${LEAD_SIM:-06}"
SEEDS=(${LEAD_SEEDS:-1 2 3 4 5})
BUDGET="${MAX_ORACLE_CALLS:-10000}"
CONFIG="${CHEML_CONFIG:-genetic_chemalactica/genetic/configs/llama3_3b_scale.yaml}"
ENV_NAME="${ORACLE_CONDA_ENV:-cheml}"

ORACLE_HOST="${ORACLE_HOST:-127.0.0.1}"
ORACLE_PORTS=(${ORACLE_PORTS:-5050 5051 5052})
ORACLE_THREADS="${ORACLE_THREADS:-80}"
MAX_CONCURRENT_DOCKING="${MAX_CONCURRENT_DOCKING:-80}"
NUM_CPU_DOCK="${NUM_CPU_DOCK:-1}"

# Shared GPU pool: genetic_runner round-robins seeds onto cuda:0..4, so
# RNG seed 1 of every seed-molecule lands on the first listed GPU.
GPUS="${GPU_GROUP:-0,1,2,3,4}"
N_GPUS="$(awk -F, '{print NF}' <<<"$GPUS")"

if [[ ${#SEEDMOLS[@]} -ne ${#ORACLE_PORTS[@]} ]]; then
  echo "ERROR: need one oracle port per seed molecule (${#SEEDMOLS[@]} mols, ${#ORACLE_PORTS[@]} ports)" >&2
  exit 1
fi

task_name() {
  local seedmol="$1"
  echo "lead.${TARGET}_${SIM}_${seedmol}"
}

LOG_ROOT="${OUT_DIR}/launch_logs/lead_wave_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

oracle_healthy() {
  local url="$1"
  curl -fsS --max-time 5 "${url}/health" >/dev/null 2>&1
}

start_oracle_if_needed() {
  local port="$1" log="$2"
  local url="http://${ORACLE_HOST}:${port}"
  if oracle_healthy "$url"; then
    echo "Oracle already healthy on ${url}"
    return 0
  fi
  echo "Starting Vina oracle on ${url} (${MAX_CONCURRENT_DOCKING} concurrent docks, ${NUM_CPU_DOCK} CPU/dock)"
  NUM_CPU_DOCK="$NUM_CPU_DOCK" \
  MAX_CONCURRENT_DOCKING="$MAX_CONCURRENT_DOCKING" \
  ORACLE_THREADS="$ORACLE_THREADS" \
  ORACLE_PORT="$port" \
  ORACLE_HOST="$ORACLE_HOST" \
  OUT_DIR="$OUT_DIR" \
  nohup "${PROJECT_ROOT}/benchmark/docking_oracle/start_oracle.sh" \
    >"$log" 2>&1 &
  echo $! >"${log}.pid"
}

wait_oracle() {
  local url="$1" name="$2"
  echo "Waiting for ${name} at ${url}"
  local i
  for i in $(seq 1 60); do
    if oracle_healthy "$url"; then
      echo "${name} healthy"
      curl -fsS "${url}/health" || true
      echo
      return 0
    fi
    sleep 2
  done
  echo "ERROR: ${name} at ${url} did not become healthy" >&2
  return 1
}

run_group() {
  local task="$1" url="$2" log="$3"
  echo "Launching ${task} seeds ${SEEDS[*]} → ${url} GPUs=${GPUS} (seed i → GPU i-1)"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHONPATH="$PYTHONPATH" \
  OUT_DIR="$OUT_DIR" \
  nohup conda run --no-capture-output -n "$ENV_NAME" \
    python "${PROJECT_ROOT}/genetic_chemalactica/genetic_runner.py" \
      --config_file "$CONFIG" \
      --task_name "$task" \
      --seeds "${SEEDS[@]}" \
      --max_oracle_calls "$BUDGET" \
      --reward_type hit \
      --vina_url "$url" \
      --max_workers "${#SEEDS[@]}" \
      --n_gpus "$N_GPUS" \
    >"$log" 2>&1 &
  echo $! >"${log}.pid"
}

for i in "${!ORACLE_PORTS[@]}"; do
  port="${ORACLE_PORTS[$i]}"
  start_oracle_if_needed "$port" "${LOG_ROOT}/oracle_${port}.log"
done
for i in "${!ORACLE_PORTS[@]}"; do
  port="${ORACLE_PORTS[$i]}"
  wait_oracle "http://${ORACLE_HOST}:${port}" "oracle-${i}"
done

TASKS=()
for i in "${!SEEDMOLS[@]}"; do
  task="$(task_name "${SEEDMOLS[$i]}")"
  TASKS+=("$task")
  url="http://${ORACLE_HOST}:${ORACLE_PORTS[$i]}"
  run_group "$task" "$url" "${LOG_ROOT}/${task}.log"
  # Avoid get_log_dir exp-N race between runners.
  if [[ $i -lt $((${#SEEDMOLS[@]} - 1)) ]]; then
    sleep 3
  fi
done

echo
echo "Wave started: target=${TARGET} sim=${SIM} budget=${BUDGET} gpus=${GPUS}"
echo "  RNG seed 1→GPU 0, seed 2→GPU 1, ... seed 5→GPU 4 (all 3 seed molecules share that GPU)"
for i in "${!SEEDMOLS[@]}"; do
  task="${TASKS[$i]}"
  url="http://${ORACLE_HOST}:${ORACLE_PORTS[$i]}"
  echo "  ${task}  vina=${url}  log=${LOG_ROOT}/${task}.log  pid=$(cat "${LOG_ROOT}/${task}.log.pid")"
done
echo "Launch log dir: ${LOG_ROOT}"
