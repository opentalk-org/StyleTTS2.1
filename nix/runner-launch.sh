set -euo pipefail

runner_id="${RUNNER_ID:-runner-1}"
nats_url="${NATS_URL:-nats://127.0.0.1:4222}"

echo "Starting runner $runner_id"
exec python -m runner.cli --runner-id "$runner_id" --nats-url "$nats_url"
