# Joins the Headscale tailnet in userspace mode, derives the hub's DB/NATS/S3
# endpoints from RUNFLOW_HUB_HOST, then launches the runner under proxychains-ng
# so its raw-TCP Postgres/NATS clients (and boto3) transparently route through
# the tailnet SOCKS5 proxy. No app code changes required.

# Each Salad replica must have a distinct RUNNER_ID: it keys the runner's DB
# heartbeat row (runner/heartbeat.py) and NATS work routing, so a shared id
# would make replicas overwrite each other. Prefer an explicit RUNNER_ID, else
# Salad's per-replica machine id, else the container hostname, else a default.
if [ -z "${RUNNER_ID:-}" ]; then
  if [ -n "${SALAD_MACHINE_ID:-}" ]; then
    RUNNER_ID="salad-${SALAD_MACHINE_ID}"
  else
    RUNNER_ID="runner-$(hostname 2>/dev/null || echo 1)"
  fi
fi
export RUNNER_ID

export RUNFLOW_HUB_HOST="${RUNFLOW_HUB_HOST:-runflow-hub}"
export NATS_PORT="${NATS_PORT:-4222}"
export PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
export POSTGRES_USER="${POSTGRES_USER:-runflow}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-runflow}"
export POSTGRES_DB="${POSTGRES_DB:-runflow}"
export RUSTFS_ACCESS_KEY="${RUSTFS_ACCESS_KEY:-runflow}"
export RUSTFS_SECRET_KEY="${RUSTFS_SECRET_KEY:-runflow-secret}"
export RUSTFS_BUCKET="${RUSTFS_BUCKET:-runflow}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

export TAILSCALE_HOSTNAME="${TAILSCALE_HOSTNAME:-runflow-runner-$RUNNER_ID}"
export TAILSCALE_USERSPACE="${TAILSCALE_USERSPACE:-1}"
tailscale-up

# (resolved remotely by the SOCKS5 proxy when running under proxychains).
export NATS_URL="${NATS_URL:-nats://$RUNFLOW_HUB_HOST:$NATS_PORT}"
export RUNFLOW_PGBOUNCER_DATABASE_URL="${RUNFLOW_PGBOUNCER_DATABASE_URL:-postgresql+psycopg://$POSTGRES_USER:$POSTGRES_PASSWORD@$RUNFLOW_HUB_HOST:$PGBOUNCER_PORT/$POSTGRES_DB}"
export RUNFLOW_S3_ENDPOINT_URL="${RUNFLOW_S3_ENDPOINT_URL:-http://$RUNFLOW_HUB_HOST:9000}"
export RUNFLOW_S3_BUCKET="${RUNFLOW_S3_BUCKET:-$RUSTFS_BUCKET}"
export RUNFLOW_S3_REGION="${RUNFLOW_S3_REGION:-$AWS_REGION}"
export RUNFLOW_S3_ACCESS_KEY_ID="${RUNFLOW_S3_ACCESS_KEY_ID:-$RUSTFS_ACCESS_KEY}"
export RUNFLOW_S3_SECRET_ACCESS_KEY="${RUNFLOW_S3_SECRET_ACCESS_KEY:-$RUSTFS_SECRET_KEY}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-$RUSTFS_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-$RUSTFS_SECRET_KEY}"
export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-$RUNFLOW_S3_ENDPOINT_URL}"

# Writable cache dirs on the /data volume (root has no home in the minimal image).
mkdir -p "${HOME:?}" "${XDG_CACHE_HOME:?}" "${HF_HOME:?}" "${TORCHINDUCTOR_CACHE_DIR:?}"

echo "runner-entrypoint: runner=$RUNNER_ID hub=$RUNFLOW_HUB_HOST nats=$NATS_URL"

# Only route through the SOCKS5 proxy when userspace tailscale actually brought
# it up. In kernel mode, or standalone (no authkey, e.g. hub-on-same-host test),
# connect directly — there is no proxy on :1055.
if [ "$TAILSCALE_USERSPACE" = "1" ] && [ -n "${TAILSCALE_AUTHKEY:-}" ]; then
  proxychains_conf=/tmp/proxychains-runner.conf
  cat > "$proxychains_conf" <<'EOF'
strict_chain
proxy_dns
remote_dns_subnet 224
tcp_read_time_out 15000
tcp_connect_time_out 8000
[ProxyList]
socks5 127.0.0.1 1055
EOF
  echo "runner-entrypoint: launching runner via proxychains (tailnet SOCKS5)"
  exec proxychains4 -f "$proxychains_conf" runflow-runner-launch
else
  echo "runner-entrypoint: launching runner directly (no userspace SOCKS proxy)"
  exec runflow-runner-launch
fi
