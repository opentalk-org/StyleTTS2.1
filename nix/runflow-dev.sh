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
export BACKEND_PORT="${BACKEND_PORT:-8001}"
export VITE_BACKEND_URL="${VITE_BACKEND_URL:-http://$BACKEND_HOST:$BACKEND_PORT}"
export FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export RUNNER_ID="${RUNNER_ID:-runner-1}"
export AIM_REPO="${AIM_REPO:-.data/aim}"
export AIM_HOST="${AIM_HOST:-127.0.0.1}"
export AIM_PORT="${AIM_PORT:-43800}"
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
export PYTHONPATH="$PWD/src"

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

if [ "$(id -u)" = "0" ]; then
  runflow_dev_user="${RUNFLOW_DEV_USER:-user}"
  if ! id "$runflow_dev_user" >/dev/null 2>&1; then
    echo "runflow-dev cannot run PostgreSQL as root, and RUNFLOW_DEV_USER '$runflow_dev_user' does not exist" >&2
    exit 1
  fi

  mkdir -p "$NATS_DATA" "$RUSTFS_DATA" "$PGDATA" "$PGHOST" "$pgbouncer_dir"
  chown -R "$runflow_dev_user" "$PWD/.data"
  chmod 700 "$PGDATA" "$PGHOST" "$pgbouncer_dir"
  if [ -d "$PWD/src/frontend/node_modules" ]; then
    chown -R "$runflow_dev_user" "$PWD/src/frontend/node_modules"
  fi
  if [ -d "$PWD/.venv" ]; then
    chown -R "$runflow_dev_user" "$PWD/.venv"
  fi

  echo "runflow-dev cannot run PostgreSQL as root; re-executing as $runflow_dev_user"
  user_home="$(getent passwd "$runflow_dev_user" | cut -d: -f6)"
  user_tmp="${TMPDIR:-/tmp}"
  case "$user_tmp" in
    /tmp/nix-shell.*) user_tmp="/tmp" ;;
  esac
  if [ ! -d "$user_tmp" ] || [ ! -w "$user_tmp" ]; then
    user_tmp="/tmp"
  fi
  mkdir -p "$user_tmp" "$user_home/.cache" "$user_home/.local/share" "$user_home/.local/state"
  chown "$runflow_dev_user" "$user_home/.cache" "$user_home/.local" "$user_home/.local/share" "$user_home/.local/state" 2>/dev/null || true
  exec runuser -u "$runflow_dev_user" -- env \
    HOME="$user_home" \
    TMPDIR="$user_tmp" \
    XDG_CACHE_HOME="$user_home/.cache" \
    XDG_DATA_HOME="$user_home/.local/share" \
    XDG_STATE_HOME="$user_home/.local/state" \
    "$0" "$@"
fi

mkdir -p "$NATS_DATA" "$RUSTFS_DATA" "$PGDATA" "$PGHOST" "$pgbouncer_dir"
chmod 700 "$PGDATA" "$PGHOST" "$pgbouncer_dir"

if [ ! -f uv.lock ]; then
  echo "uv.lock is missing; run uv lock before starting runflow-dev" >&2
  exit 1
fi

venv_sync_stamp=".venv/.runflow-uv-sync-stamp"
if [ ! -x .venv/bin/python ] || [ ! -x .venv/bin/uvicorn ] || [ pyproject.toml -nt "$venv_sync_stamp" ] || [ uv.lock -nt "$venv_sync_stamp" ]; then
  echo "Syncing Python environment from uv.lock"
  uv sync --frozen
  touch "$venv_sync_stamp"
fi

# shellcheck disable=SC1091
. .venv/bin/activate

# Kill any stale process still listening on a port we're about to bind. Prior dev
# runs can leave app-tier services orphaned (e.g. an Aim UI double-forked past the
# shutdown trap); a fresh run then fails to bind, exits, and the wait -n below tears
# the whole stack down. Only used for services that have no "reuse existing" guard
# (backend/frontend/aim) -- Postgres/PgBouncer/NATS/RustFS are detected and reused.
free_port() {
  local port="$1" label="${2:-port}" pids
  # `|| true` inside the substitution: when the port is free, grep matches nothing
  # and exits non-zero, which under `set -o pipefail`/`set -e` would otherwise abort
  # the whole script on a standalone assignment.
  pids="$(ss -tlnpH 2>/dev/null \
    | awk -v p=":$port\$" '$4 ~ p' \
    | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u || true)"
  [ -n "$pids" ] || return 0
  echo "Freeing $label port $port from stale process(es): $(echo "$pids" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    ss -tlnpH 2>/dev/null | awk -v p=":$port\$" '$4 ~ p' | grep -q . || return 0
    sleep 1
  done
  # shellcheck disable=SC2046
  kill -9 $(ss -tlnpH 2>/dev/null | awk -v p=":$port\$" '$4 ~ p' \
    | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | sort -u) 2>/dev/null || true
}

pid_postgres=""
pid_pgbouncer=""
pid_nats=""
pid_rustfs=""
pid_backend=""
pid_frontend=""
pid_runners=""
pid_aim=""

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

if pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
  echo "Using existing PostgreSQL at $PGHOST:$PGPORT"
else
  rm -f "$PGHOST/.s.PGSQL.$PGPORT" "$PGHOST/.s.PGSQL.$PGPORT.lock"
  echo "Starting PostgreSQL"
  pg_ctl -D "$PGDATA" \
    -o "-k $PGHOST -p $PGPORT" \
    -l .data/postgres.log \
    start
  pid_postgres=started
fi

until pg_isready -h "$PGHOST" -p "$PGPORT" -d postgres; do
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

if pg_isready -h 127.0.0.1 -p "$PGBOUNCER_PORT" -d "$POSTGRES_DB" -U "$POSTGRES_USER" >/dev/null 2>&1; then
  echo "Using existing PgBouncer on port $PGBOUNCER_PORT"
else
  echo "Starting PgBouncer on port $PGBOUNCER_PORT"
  pgbouncer "$pgbouncer_config" > .data/pgbouncer.log 2>&1 &
  pid_pgbouncer=$!
fi

until pg_isready -h 127.0.0.1 -p "$PGBOUNCER_PORT" -d "$POSTGRES_DB"; do
  if [ -n "$pid_pgbouncer" ] && ! kill -0 "$pid_pgbouncer" 2>/dev/null; then
    echo "PgBouncer exited before becoming ready"
    cat .data/pgbouncer.log
    exit 1
  fi
  sleep 1
done

if python -c "import socket; socket.create_connection(('127.0.0.1', $NATS_PORT), 1).close()" >/dev/null 2>&1; then
  echo "Using existing NATS on $NATS_URL"
else
  echo "Starting NATS JetStream on $NATS_URL"
  nats-server -js -sd "$NATS_DATA" -p "$NATS_PORT" &
  pid_nats=$!
fi

until python -c "import socket; socket.create_connection(('127.0.0.1', $NATS_PORT), 1).close()" >/dev/null 2>&1; do
  if [ -n "$pid_nats" ] && ! kill -0 "$pid_nats" 2>/dev/null; then
    echo "NATS exited before becoming ready"
    exit 1
  fi
  sleep 1
done
if [ -n "$pid_nats" ] && ! kill -0 "$pid_nats" 2>/dev/null; then
  echo "NATS exited before becoming ready"
  exit 1
fi

if aws --endpoint-url "$AWS_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; then
  echo "Using existing RustFS at $AWS_ENDPOINT_URL"
else
  echo "Starting RustFS at $AWS_ENDPOINT_URL"
  rustfs > /tmp/runflow-rustfs.log 2>&1 &
  pid_rustfs=$!
fi

until aws --endpoint-url "$AWS_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; do
  if [ -n "$pid_rustfs" ] && ! kill -0 "$pid_rustfs" 2>/dev/null; then
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

echo "Legacy static UI is available at http://$BACKEND_HOST:$BACKEND_PORT/ui-old"
echo "Starting backend API at http://$BACKEND_HOST:$BACKEND_PORT"
# Auto-reload backend + runner on source changes so node edits (and the node schema
# the UI reads) take effect without a manual restart. Disable with RUNFLOW_RELOAD=0.
export RUNFLOW_RELOAD="${RUNFLOW_RELOAD:-1}"
backend_reload_args=()
if [ "$RUNFLOW_RELOAD" = "1" ]; then
  backend_reload_args=(--reload --reload-dir src)
fi
free_port "$BACKEND_PORT" backend
uvicorn backend.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" "${backend_reload_args[@]}" &
pid_backend=$!

until python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:$BACKEND_PORT/health', timeout=1).read()" >/dev/null 2>&1; do
  if [ -n "$pid_backend" ] && ! kill -0 "$pid_backend" 2>/dev/null; then
    echo "Backend exited before becoming ready"
    exit 1
  fi
  sleep 1
done

mkdir -p "$AIM_REPO"
if [ ! -d "$AIM_REPO/.aim" ]; then
  aim init --repo "$AIM_REPO" || echo "aim init failed; continuing without Aim UI"
fi
echo "Starting Aim UI at http://$AIM_HOST:$AIM_PORT"
free_port "$AIM_PORT" aim
aim up --repo "$AIM_REPO" --host "$AIM_HOST" --port "$AIM_PORT" > .data/aim.log 2>&1 &
pid_aim=$!

echo "Starting frontend at http://$FRONTEND_HOST:$FRONTEND_PORT"
free_port "$FRONTEND_PORT" frontend
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
  [ -z "$pid_aim" ] || kill "$pid_aim" 2>/dev/null || true
  [ -z "$pid_runners" ] || kill "$pid_runners" 2>/dev/null || true
  [ -z "$pid_frontend" ] || kill "$pid_frontend" 2>/dev/null || true
  [ -z "$pid_backend" ] || kill "$pid_backend" 2>/dev/null || true
  sleep 1
  [ -z "$pid_rustfs" ] || kill "$pid_rustfs" 2>/dev/null || true
  [ -z "$pid_nats" ] || kill "$pid_nats" 2>/dev/null || true
  [ -z "$pid_pgbouncer" ] || kill "$pid_pgbouncer" 2>/dev/null || true
  [ -z "$pid_postgres" ] || pg_ctl -D "$PGDATA" stop -m fast >/dev/null 2>&1 || true
}

trap shutdown EXIT INT TERM
wait_pids=()
[ -z "$pid_backend" ] || wait_pids+=("$pid_backend")
[ -z "$pid_frontend" ] || wait_pids+=("$pid_frontend")
[ -z "$pid_runners" ] || wait_pids+=("$pid_runners")
[ -z "$pid_rustfs" ] || wait_pids+=("$pid_rustfs")
[ -z "$pid_nats" ] || wait_pids+=("$pid_nats")
[ -z "$pid_pgbouncer" ] || wait_pids+=("$pid_pgbouncer")
wait -n "${wait_pids[@]}"
