set -euo pipefail

export NATS_DATA="${NATS_DATA:-.data/nats}"
export NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
export BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export RUNNER_ID="${RUNNER_ID:-runner-1}"
export RUSTFS_DATA="${RUSTFS_DATA:-.data/rustfs}"
export RUSTFS_VOLUMES="${RUSTFS_VOLUMES:-$RUSTFS_DATA}"
export RUSTFS_ADDRESS="${RUSTFS_ADDRESS:-127.0.0.1:9000}"
export RUSTFS_CONSOLE_ENABLE="${RUSTFS_CONSOLE_ENABLE:-true}"
export RUSTFS_CONSOLE_ADDRESS="${RUSTFS_CONSOLE_ADDRESS:-127.0.0.1:9001}"
export RUSTFS_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-runflow}"
export RUSTFS_SECRET_KEY="${RUSTFS_SECRET_KEY:-runflow-secret}"
export RUSTFS_BUCKET="${RUSTFS_BUCKET:-runflow}"
export AWS_ACCESS_KEY_ID="$RUSTFS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_SECRET_KEY"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://$RUSTFS_ADDRESS}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p "$NATS_DATA" "$RUSTFS_DATA"

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

echo "Starting RustFS at $AWS_ENDPOINT_URL"
rustfs > /tmp/runflow-rustfs.log 2>&1 &
pid_rustfs=$!

until aws --endpoint-url "$AWS_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; do
  if ! kill -0 "$pid_rustfs" 2>/dev/null; then
    echo "RustFS exited before becoming ready"
    cat /tmp/runflow-rustfs.log
    exit 1
  fi
  sleep 1
done

if ! aws --endpoint-url "$AWS_ENDPOINT_URL" s3api head-bucket \
  --bucket "$RUSTFS_BUCKET" >/dev/null 2>&1; then
  echo "Creating RustFS bucket $RUSTFS_BUCKET"
  aws --endpoint-url "$AWS_ENDPOINT_URL" s3api create-bucket \
    --bucket "$RUSTFS_BUCKET" >/dev/null
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
  kill "$pid_runner" "$pid_frontend" "$pid_backend" "$pid_rustfs" "$pid_nats" 2>/dev/null || true
}

trap shutdown EXIT INT TERM
wait -n "$pid_backend" "$pid_frontend" "$pid_runner" "$pid_rustfs" "$pid_nats"
