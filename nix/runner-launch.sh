set -euo pipefail

runner_count="${RUNFLOW_RUNNER_COUNT:-}"
if [ -z "$runner_count" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    runner_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  else
    runner_count=0
  fi
fi

if [ "$runner_count" -lt 1 ]; then
  runner_count=1
fi

pids=()
index=0
while [ "$index" -lt "$runner_count" ]; do
  runner_id="${RUNNER_ID_PREFIX:-runner-gpu}-$index"
  echo "Starting runner $runner_id"
  (
    export RUNNER_ID="$runner_id"
    export RUNFLOW_RUNNER_GPU_INDEX="$index"
    if command -v nvidia-smi >/dev/null 2>&1; then
      export CUDA_VISIBLE_DEVICES="$index"
    fi
    python -m runner.cli --runner-id "$runner_id" --nats-url "$NATS_URL"
  ) &
  pids+=("$!")
  index=$((index + 1))
done

shutdown() {
  kill "${pids[@]}" 2>/dev/null || true
}

trap shutdown EXIT INT TERM
wait "${pids[@]}"
