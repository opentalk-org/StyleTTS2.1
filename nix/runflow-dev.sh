set -euo pipefail

export PGDATA="${PGDATA:-.data/postgres}"
export PGHOST="${PGHOST:-.data/postgres-socket}"
export PGPORT="${PGPORT:-5432}"
export PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
export POSTGRES_DB="${POSTGRES_DB:-runflow}"
export POSTGRES_USER="${POSTGRES_USER:-runflow}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-runflow}"
export MLFLOW_DATABASE="${MLFLOW_DATABASE:-mlflow}"
export MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
export MLFLOW_PORT="${MLFLOW_PORT:-7860}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:$MLFLOW_PORT}"
export BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8001}"
export VITE_BACKEND_URL="${VITE_BACKEND_URL:-http://$BACKEND_HOST:$BACKEND_PORT}"
export FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
export FRONTEND_PORT="${FRONTEND_PORT:-5173}"
export RUNNER_ID="${RUNNER_ID:-runner-1}"
export RCLONE_S3_HOST="${RCLONE_S3_HOST:-127.0.0.1}"
export RCLONE_S3_PORT="${RCLONE_S3_PORT:-8002}"
export RCLONE_S3_PATH="${RCLONE_S3_PATH:-/home/storagebucket}"
export RCLONE_S3_SSH="${RCLONE_S3_SSH:-ssh hetzner-storagebox}"
export BUCKET_DATA="${BUCKET_DATA:-.data/rustfs}"
export BUCKET_VOLUMES="${BUCKET_VOLUMES:-$BUCKET_DATA}"
export BUCKET_ADDRESS="${BUCKET_ADDRESS:-127.0.0.1:9000}"
export BUCKET_CONSOLE_ENABLE="${BUCKET_CONSOLE_ENABLE:-true}"
export BUCKET_CONSOLE_ADDRESS="${BUCKET_CONSOLE_ADDRESS:-127.0.0.1:9001}"
export BUCKET_ACCESS_KEY="${BUCKET_ACCESS_KEY:-runflow}"
export BUCKET_SECRET_KEY="${BUCKET_SECRET_KEY:-runflow-secret}"
export BUCKET_NAME="${BUCKET_NAME:-runflow}"
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET_KEY"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://$BUCKET_ADDRESS}"
export RUNFLOW_S3_BUCKET="${RUNFLOW_S3_BUCKET:-$BUCKET_NAME}"
export RUNFLOW_S3_ENDPOINT_URL="${RUNFLOW_S3_ENDPOINT_URL:-$AWS_ENDPOINT_URL}"
export RUNFLOW_S3_REGION="${RUNFLOW_S3_REGION:-$AWS_REGION}"
export RUNFLOW_S3_ACCESS_KEY_ID="${RUNFLOW_S3_ACCESS_KEY_ID:-$BUCKET_ACCESS_KEY}"
export RUNFLOW_S3_SECRET_ACCESS_KEY="${RUNFLOW_S3_SECRET_ACCESS_KEY:-$BUCKET_SECRET_KEY}"
export RUNFLOW_PGBOUNCER_DATABASE_URL="postgresql+psycopg://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGBOUNCER_PORT/$POSTGRES_DB"
export RUNFLOW_NOTIFY_DATABASE_URL="postgresql+psycopg://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGPORT/$POSTGRES_DB"
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

  mkdir -p "$BUCKET_DATA" "$PGDATA" "$PGHOST" "$pgbouncer_dir"
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

sync_frontend_dependencies "$PWD/src/frontend"

mkdir -p "$BUCKET_DATA" "$PGDATA" "$PGHOST" "$pgbouncer_dir"
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
# runs can leave app-tier services orphaned (e.g. an MLflow UI worker past the
# shutdown trap); a fresh run then fails to bind, exits, and the wait -n below tears
# the whole stack down. Only used for services that have no "reuse existing" guard
# (backend/frontend/mlflow) -- Postgres/PgBouncer/RustFS are detected and reused.
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
pid_rustfs=""
pid_backend=""
pid_frontend=""
pid_runners=""
pid_mlflow=""
pid_rclone_s3=""

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

if ! psql -h "$PGHOST" -p "$PGPORT" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '$MLFLOW_DATABASE'" \
  | grep -q 1; then
  createdb -h "$PGHOST" -p "$PGPORT" -O "$POSTGRES_USER" "$MLFLOW_DATABASE"
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

if aws --endpoint-url "$AWS_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; then
  echo "Using existing RustFS at $AWS_ENDPOINT_URL"
else
  echo "Starting RustFS at $AWS_ENDPOINT_URL"
  rustfs_args=(
    server
    --address "$BUCKET_ADDRESS"
    --access-key "$BUCKET_ACCESS_KEY"
    --secret-key "$BUCKET_SECRET_KEY"
    --console-address "$BUCKET_CONSOLE_ADDRESS"
    "$BUCKET_VOLUMES"
  )
  if [ "$BUCKET_CONSOLE_ENABLE" = "true" ]; then
    rustfs_args+=(--console-enable)
  fi
  rustfs "${rustfs_args[@]}" > /tmp/runflow-rustfs.log 2>&1 &
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
  --bucket "$BUCKET_NAME" >/dev/null 2>&1; then
  echo "Creating RustFS bucket $BUCKET_NAME"
  aws --endpoint-url "$AWS_ENDPOINT_URL" s3api create-bucket \
    --bucket "$BUCKET_NAME" >/dev/null
fi

# Recursively TERM a process and all its descendants (children first), so killing a
# supervised service also stops its reload workers / spawned subprocesses.
kill_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_tree "$kid"
  done
  kill "$pid" 2>/dev/null || true
}

# Keep a service alive: (re)launch the command and, whenever it exits, restart it
# after a short delay instead of letting a crash take the whole dev stack down.
# Backgrounded as a subshell whose pid we track; on shutdown we TERM the subshell,
# whose trap kills the currently-running child (and its descendants). Optional
# `--port N` frees a stale listener before each (re)spawn so a crashed reload-worker
# can't block the restart. Disable restart-on-crash with RUNFLOW_RESTART=0.
supervise() {
  local label="$1"; shift
  local port=""
  if [ "$1" = "--port" ]; then port="$2"; shift 2; fi
  local delay="${RUNFLOW_RESTART_DELAY:-2}"
  local child=""
  trap 'trap - TERM INT; [ -n "$child" ] && kill_tree "$child"; exit 0' TERM INT
  local first=1
  while true; do
    if [ "$first" = 0 ] && [ -n "$port" ]; then
      free_port "$port" "$label"
    fi
    first=0
    "$@" &
    child=$!
    wait "$child" || true
    if [ "${RUNFLOW_RESTART:-1}" != "1" ]; then
      echo "[$label] exited; restart-on-crash disabled (RUNFLOW_RESTART=0)" >&2
      return 0
    fi
    echo "[$label] exited; restarting in ${delay}s" >&2
    sleep "$delay"
  done
}

echo "Starting rclone S3 at http://$RCLONE_S3_HOST:$RCLONE_S3_PORT"
free_port "$RCLONE_S3_PORT" rclone-s3
supervise rclone-s3 --port "$RCLONE_S3_PORT" rclone serve s3 \
  --addr "$RCLONE_S3_HOST:$RCLONE_S3_PORT" \
  --auth-key "$BUCKET_ACCESS_KEY,$BUCKET_SECRET_KEY" \
  --sftp-ssh "$RCLONE_S3_SSH" \
  ":sftp:$RCLONE_S3_PATH" &
pid_rclone_s3=$!

echo "Legacy static UI is available at http://$BACKEND_HOST:$BACKEND_PORT/ui-old"
echo "Starting backend API at http://$BACKEND_HOST:$BACKEND_PORT"
# Auto-reload backend + runner on source changes so node edits (and the node schema
# the UI reads) take effect without a manual restart. Disable with RUNFLOW_RELOAD=0.
# Both are also supervised (see supervise()), so a crash is restarted automatically.
export RUNFLOW_RELOAD="${RUNFLOW_RELOAD:-1}"
backend_reload_args=()
if [ "$RUNFLOW_RELOAD" = "1" ]; then
  backend_reload_args=(--reload --reload-dir src)
fi
free_port "$BACKEND_PORT" backend
supervise backend --port "$BACKEND_PORT" uvicorn backend.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" "${backend_reload_args[@]}" &
pid_backend=$!

# pid_backend is the supervisor, which stays up across backend restarts. Bound the
# wait so a crash-looping backend prints its errors instead of hanging startup here.
backend_wait=0
until python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:$BACKEND_PORT/health', timeout=1).read()" >/dev/null 2>&1; do
  if [ -n "$pid_backend" ] && ! kill -0 "$pid_backend" 2>/dev/null; then
    echo "Backend supervisor exited before becoming ready"
    exit 1
  fi
  backend_wait=$((backend_wait + 1))
  if [ "$backend_wait" -ge 120 ]; then
    echo "Backend not healthy after ${backend_wait}s; continuing (supervisor keeps retrying)"
    break
  fi
  sleep 1
done

export MLFLOW_BACKEND_STORE_URI="postgresql+psycopg://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGPORT/$MLFLOW_DATABASE"
export MLFLOW_ARTIFACTS_DESTINATION="s3://$BUCKET_NAME/mlflow"
export MLFLOW_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"
echo "Starting MLflow UI at http://$MLFLOW_HOST:$MLFLOW_PORT"
free_port "$MLFLOW_PORT" mlflow
supervise mlflow --port "$MLFLOW_PORT" mlflow server \
  --host "$MLFLOW_HOST" \
  --port "$MLFLOW_PORT" \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
  --allowed-hosts "localhost:*,127.0.0.1:*" \
  --x-frame-options NONE > .data/mlflow.log 2>&1 &
pid_mlflow=$!

mlflow_wait=0
until python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:$MLFLOW_PORT/health', timeout=1).read()" >/dev/null 2>&1; do
  if ! kill -0 "$pid_mlflow" 2>/dev/null; then
    echo "MLflow supervisor exited before becoming ready"
    exit 1
  fi
  mlflow_wait=$((mlflow_wait + 1))
  if [ "$mlflow_wait" -ge 120 ]; then
    echo "MLflow not healthy after ${mlflow_wait}s"
    cat .data/mlflow.log
    exit 1
  fi
  sleep 1
done

echo "Starting frontend at http://$FRONTEND_HOST:$FRONTEND_PORT"
free_port "$FRONTEND_PORT" frontend
(
  cd src/frontend
  exec npm run dev -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
pid_frontend=$!

echo "Starting runners"
supervise runner bash nix/runner-launch.sh &
pid_runners=$!

shutdown() {
  echo "Stopping Runflow dev services"
  [ -z "$pid_mlflow" ] || kill "$pid_mlflow" 2>/dev/null || true
  [ -z "$pid_runners" ] || kill "$pid_runners" 2>/dev/null || true
  [ -z "$pid_frontend" ] || kill "$pid_frontend" 2>/dev/null || true
  [ -z "$pid_backend" ] || kill "$pid_backend" 2>/dev/null || true
  [ -z "$pid_rclone_s3" ] || kill "$pid_rclone_s3" 2>/dev/null || true
  sleep 1
  [ -z "$pid_rustfs" ] || kill "$pid_rustfs" 2>/dev/null || true
  [ -z "$pid_pgbouncer" ] || kill "$pid_pgbouncer" 2>/dev/null || true
  [ -z "$pid_postgres" ] || pg_ctl -D "$PGDATA" stop -m fast >/dev/null 2>&1 || true
}

trap shutdown EXIT INT TERM
wait_pids=()
[ -z "$pid_backend" ] || wait_pids+=("$pid_backend")
[ -z "$pid_mlflow" ] || wait_pids+=("$pid_mlflow")
[ -z "$pid_frontend" ] || wait_pids+=("$pid_frontend")
[ -z "$pid_runners" ] || wait_pids+=("$pid_runners")
[ -z "$pid_rclone_s3" ] || wait_pids+=("$pid_rclone_s3")
[ -z "$pid_rustfs" ] || wait_pids+=("$pid_rustfs")
[ -z "$pid_pgbouncer" ] || wait_pids+=("$pid_pgbouncer")
wait -n "${wait_pids[@]}"
