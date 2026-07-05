set -euo pipefail

export PGDATA="${PGDATA:-/data/postgres}"
export PGHOST="${PGHOST:-/tmp/postgres}"
export PGPORT="${PGPORT:-5432}"
export PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
export POSTGRES_DB="${POSTGRES_DB:-runflow}"
export POSTGRES_USER="${POSTGRES_USER:-runflow}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-runflow}"
export NATS_DATA="${NATS_DATA:-/data/nats}"
export NATS_URL="${NATS_URL:-nats://127.0.0.1:4222}"
export RUSTFS_DATA="${RUSTFS_DATA:-/data/rustfs}"
export RUSTFS_VOLUMES="${RUSTFS_VOLUMES:-$RUSTFS_DATA}"
export RUSTFS_ADDRESS="${RUSTFS_ADDRESS:-0.0.0.0:9000}"
export RUSTFS_CONSOLE_ENABLE="${RUSTFS_CONSOLE_ENABLE:-true}"
export RUSTFS_CONSOLE_ADDRESS="${RUSTFS_CONSOLE_ADDRESS:-0.0.0.0:9001}"
export RUSTFS_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-runflow}"
export RUSTFS_SECRET_KEY="${RUSTFS_SECRET_KEY:-runflow-secret}"
export RUSTFS_BUCKET="${RUSTFS_BUCKET:-runflow}"
export AWS_ACCESS_KEY_ID="$RUSTFS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$RUSTFS_SECRET_KEY"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://127.0.0.1:9000}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGBOUNCER_PORT/$POSTGRES_DB"

pgbouncer_dir=/tmp/pgbouncer
pgbouncer_config="$pgbouncer_dir/pgbouncer.ini"
pgbouncer_userlist="$pgbouncer_dir/userlist.txt"

mkdir -p "$PGDATA" "$PGHOST" "$NATS_DATA" "$RUSTFS_DATA" "$pgbouncer_dir"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "Initializing PostgreSQL database at $PGDATA"
  initdb -D "$PGDATA" --auth=trust

  cat >> "$PGDATA/postgresql.conf" <<EOF
listen_addresses = '*'
port = $PGPORT
unix_socket_directories = '$PGHOST'
EOF

  cat >> "$PGDATA/pg_hba.conf" <<EOF
host all all 0.0.0.0/0 md5
host all all ::/0 md5
EOF
fi

echo "Starting PostgreSQL"
pg_ctl -D "$PGDATA" \
  -o "-k $PGHOST -p $PGPORT" \
  -l /tmp/postgres.log \
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
listen_addr = 0.0.0.0
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
pgbouncer "$pgbouncer_config" > /tmp/pgbouncer.log 2>&1 &
pid_pgbouncer=$!

until pg_isready -h 127.0.0.1 -p "$PGBOUNCER_PORT" -d "$POSTGRES_DB"; do
  if ! kill -0 "$pid_pgbouncer" 2>/dev/null; then
    echo "PgBouncer exited before becoming ready"
    cat /tmp/pgbouncer.log
    exit 1
  fi
  sleep 1
done

echo "Starting NATS JetStream"
nats-server -js -sd "$NATS_DATA" -p 4222 > /tmp/nats.log 2>&1 &
pid_nats=$!

until grep -q "Server is ready" /tmp/nats.log 2>/dev/null; do
  if ! kill -0 "$pid_nats" 2>/dev/null; then
    echo "NATS exited before becoming ready"
    exit 1
  fi
  sleep 1
done

echo "Starting RustFS"
rustfs > /tmp/rustfs.log 2>&1 &
pid_rustfs=$!

until aws --endpoint-url "$AWS_ENDPOINT_URL" s3api list-buckets >/dev/null 2>&1; do
  if ! kill -0 "$pid_rustfs" 2>/dev/null; then
    echo "RustFS exited before becoming ready"
    cat /tmp/rustfs.log
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

echo "Starting Runflow backend"
runflow-backend &
pid_backend=$!

echo "Starting Runflow runner"
runflow-runner &
pid_runner=$!

shutdown() {
  echo "Stopping services"
  kill "$pid_backend" "$pid_runner" "$pid_rustfs" "$pid_nats" "$pid_pgbouncer" 2>/dev/null || true
  pg_ctl -D "$PGDATA" stop -m fast || true
}

trap shutdown EXIT INT TERM
wait -n "$pid_backend" "$pid_runner" "$pid_rustfs" "$pid_nats" "$pid_pgbouncer"
