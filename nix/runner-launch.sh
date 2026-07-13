set -euo pipefail

runner_id="${RUNNER_ID:-runner-1}"

echo "Starting runner $runner_id"
if [ "${RUNFLOW_RELOAD:-0}" = "1" ]; then
  # Auto-restart the runner when node source changes so edits take effect without a
  # manual restart. Restarting drops any in-flight run and the loaded model cache.
  echo "Runner auto-reload enabled (watching src/)"
  exec python -m watchfiles --target-type command "python -m runner.cli --runner-id $runner_id" src
else
  exec python -m runner.cli --runner-id "$runner_id"
fi
