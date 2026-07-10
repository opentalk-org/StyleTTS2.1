# Invoked (NOT sourced) as the `tailscale-up` command by both the hub and the
# runner entrypoints. tailscaled is backgrounded and intentionally survives this
# script's exit so the parent entrypoint keeps its tailnet connectivity (and, in
# userspace mode, the SOCKS5 proxy the runner routes through).
# Modes (TAILSCALE_USERSPACE):
#   0 = kernel/TUN  (hub on Vast.ai) — real interface, local 0.0.0.0 services
#       are reachable by tailnet peers. Needs --device=/dev/net/tun + NET_ADMIN.
#   1 = userspace   (runner on Salad) — no TUN needed; exposes a SOCKS5 proxy
#       on localhost:1055 for outbound tailnet connections.

if [ -z "${TAILSCALE_AUTHKEY:-}" ]; then
  echo "tailscale-up: TAILSCALE_AUTHKEY not set — skipping tailnet join (standalone mode)"
  exit 0
fi

if [ -z "${TAILSCALE_LOGIN_SERVER:-}" ]; then
  echo "tailscale-up: TAILSCALE_LOGIN_SERVER is required when TAILSCALE_AUTHKEY is set" >&2
  exit 1
fi

state_dir="${TAILSCALE_STATE_DIR:-/data/tailscale}"
sock="/var/run/tailscale/tailscaled.sock"
ts_hostname="${TAILSCALE_HOSTNAME:-runflow-node}"
userspace="${TAILSCALE_USERSPACE:-0}"

mkdir -p "$state_dir" /var/run/tailscale /tmp
ts_log="$state_dir/tailscaled.log"

if tailscale --socket="$sock" status >/dev/null 2>&1; then
  echo "tailscale-up: tailscaled already running"
elif [ "$userspace" = "1" ]; then
  echo "tailscale-up: starting tailscaled (userspace networking, SOCKS5 :1055)"
  tailscaled \
    --state="$state_dir/tailscaled.state" \
    --socket="$sock" \
    --tun=userspace-networking \
    --socks5-server=localhost:1055 \
    --outbound-http-proxy-listen=localhost:1055 \
    >"$ts_log" 2>&1 &
else
  if [ ! -e /dev/net/tun ]; then
    echo "tailscale-up: /dev/net/tun is missing — run the container with" \
         "'--device=/dev/net/tun --cap-add=NET_ADMIN', or set TAILSCALE_USERSPACE=1" >&2
    exit 1
  fi
  echo "tailscale-up: starting tailscaled (kernel/TUN networking)"
  tailscaled \
    --state="$state_dir/tailscaled.state" \
    --socket="$sock" \
    >"$ts_log" 2>&1 &
fi

i=0
while [ "$i" -lt 30 ]; do
  if tailscale --socket="$sock" status >/dev/null 2>&1; then
    break
  fi
  i=$((i + 1))
  sleep 1
done

up_args="--login-server=$TAILSCALE_LOGIN_SERVER"
up_args="$up_args --authkey=$TAILSCALE_AUTHKEY"
up_args="$up_args --hostname=$ts_hostname"
up_args="$up_args --accept-dns=true"
if [ -n "${TAILSCALE_TAGS:-}" ]; then
  up_args="$up_args --advertise-tags=$TAILSCALE_TAGS"
fi
if [ "$userspace" != "1" ]; then
  up_args="$up_args --accept-routes=true"
fi

echo "tailscale-up: joining tailnet as '$ts_hostname' via $TAILSCALE_LOGIN_SERVER"
# shellcheck disable=SC2086
tailscale --socket="$sock" up $up_args

echo "tailscale-up: connected"
tailscale --socket="$sock" status || true

# In userspace mode the netstack does not route inbound tailnet connections to
# local listeners, so expose the requested ports with raw TCP forwarders. This
# is how the hub (no TUN on Vast.ai) publishes PgBouncer/NATS/RustFS to peers.
if [ -n "${TAILSCALE_SERVE_PORTS:-}" ]; then
  for port in $TAILSCALE_SERVE_PORTS; do
    echo "tailscale-up: serving tcp/$port over the tailnet (-> 127.0.0.1:$port)"
    tailscale --socket="$sock" serve --bg --tcp="$port" "tcp://127.0.0.1:$port" \
      || echo "tailscale-up: warning: could not configure serve for tcp/$port"
  done
fi
