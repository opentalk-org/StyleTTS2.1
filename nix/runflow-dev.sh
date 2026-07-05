set -euo pipefail

export NATS_DATA="${NATS_DATA:-.data/nats}"
export NATS_PORT="${NATS_PORT:-4222}"
export NATS_URL="${NATS_URL:-nats://127.0.0.1:$NATS_PORT}"
export PGDATA="${PGDATA:-.data/postgres}"
export PGHOST="${PGHOST:-.data/postgres-socket}"
export PGPORT="${PGPORT:-5432}"
export PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
export POSTGRES_DB="${POSTGRES_DB:-runflow}"
export POSTGRES_USER="${POSTGRES_USER:-runflow}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-runflow}"
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
export RUNFLOW_S3_BUCKET="${RUNFLOW_S3_BUCKET:-$RUSTFS_BUCKET}"
export RUNFLOW_S3_ENDPOINT_URL="${RUNFLOW_S3_ENDPOINT_URL:-$AWS_ENDPOINT_URL}"
export RUNFLOW_S3_REGION="${RUNFLOW_S3_REGION:-$AWS_REGION}"
export RUNFLOW_S3_ACCESS_KEY_ID="${RUNFLOW_S3_ACCESS_KEY_ID:-$RUSTFS_ACCESS_KEY}"
export RUNFLOW_S3_SECRET_ACCESS_KEY="${RUNFLOW_S3_SECRET_ACCESS_KEY:-$RUSTFS_SECRET_KEY}"
export RUNFLOW_PGBOUNCER_DATABASE_URL="postgresql+psycopg://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGBOUNCER_PORT/$POSTGRES_DB"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

case "$PGDATA" in
  /*) ;;
  *) PGDATA="$PWD/$PGDATA" ;;
esac

case "$PGHOST" in
  /*) ;;
  *) PGHOST="$PWD/$PGHOST" ;;
esac

pgbouncer_dir="$PWD/.data/pgbouncer"
pgbouncer_config="$pgbouncer_dir/pgbouncer.ini"
pgbouncer_userlist="$pgbouncer_dir/userlist.txt"

mkdir -p "$NATS_DATA" "$RUSTFS_DATA" "$PGDATA" "$PGHOST" "$pgbouncer_dir"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "Initializing PostgreSQL database at $PGDATA"
  initdb -D "$PGDATA" --auth=trust

  cat >> "$PGDATA/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1'
port = $PGPORT
unix_socket_directories = '$PGHOST'
EOF

  cat >> "$PGDATA/pg_hba.conf" <<EOF
host all all 127.0.0.1/32 md5
host all all ::1/128 md5
EOF
fi

echo "Starting PostgreSQL"
pg_ctl -D "$PGDATA" \
  -o "-k $PGHOST -p $PGPORT" \
  -l .data/postgres.log \
  start

until pg_isready -h "$PGHOST" -p "$PGPORT"; do
  sleep 1
done

echo "Ensuring PostgreSQL database/user exist"
psql -h "$PGHOST" -p "$PGPORT" -d postgres <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles WHERE rolname = '$POSTGRES_USER'
  ) THEN
    CREATE ROLE "$POSTGRES_USER" LOGIN PASSWORD '$POSTGRES_PASSWORD';
  END IF;
END
\$\$;
SQL

if ! psql -h "$PGHOST" -p "$PGPORT" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" \
  | grep -q 1; then
  createdb -h "$PGHOST" -p "$PGPORT" -O "$POSTGRES_USER" "$POSTGRES_DB"
fi

cat > "$pgbouncer_userlist" <<EOF
"$POSTGRES_USER" "$POSTGRES_PASSWORD"
EOF

cat > "$pgbouncer_config" <<EOF
[databases]
$POSTGRES_DB = host=$PGHOST port=$PGPORT dbname=$POSTGRES_DB

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = $PGBOUNCER_PORT
unix_socket_dir = $pgbouncer_dir
auth_type = plain
auth_file = $pgbouncer_userlist
pool_mode = transaction
max_client_conn = 200
default_pool_size = 20
reserve_pool_size = 5
ignore_startup_parameters = extra_float_digits
pidfile = $pgbouncer_dir/pgbouncer.pid
EOF

echo "Starting PgBouncer on port $PGBOUNCER_PORT"
pgbouncer "$pgbouncer_config" > .data/pgbouncer.log 2>&1 &
pid_pgbouncer=$!

until pg_isready -h 127.0.0.1 -p "$PGBOUNCER_PORT" -d "$POSTGRES_DB"; do
  if ! kill -0 "$pid_pgbouncer" 2>/dev/null; then
    echo "PgBouncer exited before becoming ready"
    cat .data/pgbouncer.log
    exit 1
  fi
  sleep 1
done

echo "Starting NATS JetStream on $NATS_URL"
nats-server -js -sd "$NATS_DATA" -p "$NATS_PORT" &
pid_nats=$!

until python -c "import socket; socket.create_connection(('127.0.0.1', $NATS_PORT), 1).close()" >/dev/null 2>&1; do
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

until python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:$BACKEND_PORT/health', timeout=1).read()" >/dev/null 2>&1; do
  if ! kill -0 "$pid_backend" 2>/dev/null; then
    echo "Backend exited before becoming ready"
    exit 1
  fi
  sleep 1
done

echo "Starting frontend at http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd src/frontend
  if [ ! -d node_modules ]; then
    npm ci
  fi
  exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
pid_frontend=$!

echo "Starting runners"
bash nix/runner-launch.sh &
pid_runners=$!

shutdown() {
  echo "Stopping Runflow dev services"
  kill "$pid_runners" "$pid_frontend" "$pid_backend" 2>/dev/null || true
  sleep 1
  kill "$pid_rustfs" "$pid_nats" "$pid_pgbouncer" 2>/dev/null || true
  pg_ctl -D "$PGDATA" stop -m fast >/dev/null 2>&1 || true
}

trap shutdown EXIT INT TERM
wait -n "$pid_backend" "$pid_frontend" "$pid_runners" "$pid_rustfs" "$pid_nats" "$pid_pgbouncer"
