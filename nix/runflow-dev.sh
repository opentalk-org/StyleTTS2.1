set -euo pipefail

export NATS_DATA="${NATS_DATA:-.data/nats}"
export NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
export BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export RUNNER_ID="${RUNNER_ID:-runner-1}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p "$NATS_DATA"

echo "Starting NATS JetStream on $NATS_URL"
nats-server -js -sd "$NATS_DATA" -p 4222 &
pid_nats=$!

until python -c 'import socket; socket.create_connection(("127.0.0.1", 4222), 1).close()' >/dev/null 2>&1; do
  if ! kill -0 "$pid_nats" 2>/dev/null; then
    echo "NATS exited before becoming ready"
    exit 1
  fi
  sleep 1
done
if ! kill -0 "$pid_nats" 2>/dev/null; then
  echo "NATS exited before becoming ready"
  exit 1
fi

echo "Starting backend API at http://$BACKEND_HOST:$BACKEND_PORT"
echo "Legacy static UI is available at http://$BACKEND_HOST:$BACKEND_PORT/ui-old"
uvicorn backend.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
pid_backend=$!

echo "Starting frontend at http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd src/frontend
  if [ ! -d node_modules ]; then
    npm ci
  fi
  exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
pid_frontend=$!

echo "Starting runner $RUNNER_ID"
python -m runner.worker --runner-id "$RUNNER_ID" --nats-url "$NATS_URL" &
pid_runner=$!

shutdown() {
  echo "Stopping Runflow dev services"
  kill "$pid_runner" "$pid_frontend" "$pid_backend" "$pid_nats" 2>/dev/null || true
}

trap shutdown EXIT INT TERM
wait -n "$pid_backend" "$pid_frontend" "$pid_runner" "$pid_nats"
