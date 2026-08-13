#!/usr/bin/env bash
set -euo pipefail

offer_id="${1:?Usage: $0 VAST_OFFER_ID}"
key="$HOME/.ssh/vastai"
github_key="$HOME/.ssh/id_ed25519"
rclone_config="$HOME/.config/rclone/rclone.conf"

test -f "$key"
test -f "$key.pub"
test -f "$github_key"
test -f "$rclone_config"
command -v vastai >/dev/null

echo "Creating Vast instance from offer $offer_id"
created="$(vastai create instance "$offer_id" \
  --image ghcr.io/dialohq/devenv:sha-e7f1719 \
  --env '-p 22:22 -p 8000:8000 -p 8080:8080' \
  --disk 200 \
  --raw \
  --args)"
instance_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])' <<<"$created")"
echo "Created instance $instance_id"
vastai attach ssh "$instance_id" "$key.pub" >/dev/null
echo "Attached $key.pub to instance $instance_id"

ssh_ready=false
control_socket="/tmp/vast-$instance_id-$$"
for _ in {1..180}; do
  if vastai show instance "$instance_id" | grep -qw running; then
    ssh_url="$(vastai ssh-url "$instance_id" 2>/dev/null || true)"
    read -r ssh_host ssh_port < <(python3 -c '
import sys
from urllib.parse import urlparse
url = urlparse(sys.stdin.read().strip())
print(url.hostname or "", url.port or "")
' <<<"$ssh_url")
    ssh_args=(-T -p "$ssh_port" -i "$key" -o BatchMode=yes -o ConnectTimeout=10 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
    remote="root@$ssh_host"
    if [[ -n "$ssh_host" && -n "$ssh_port" ]] && ssh -fN -M -S "$control_socket" "${ssh_args[@]}" "$remote" </dev/null 2>/dev/null; then
      ssh -S "$control_socket" -O exit "$remote" >/dev/null 2>&1
      ssh_ready=true
      break
    fi
  fi
  echo "Waiting for running instance and SSH"
  sleep 10
done

if [[ "$ssh_ready" != true ]]; then
  echo "Instance did not become reachable over SSH" >&2
  exit 1
fi

base64 < "$github_key" | ssh "${ssh_args[@]}" "$remote" \
  'base64 -d > /root/.ssh/id_ed25519'
base64 < "$rclone_config" | ssh "${ssh_args[@]}" "$remote" \
  'mkdir -p /root/.config/rclone && base64 -d > /root/.config/rclone/rclone.conf'
if [[ -f "$(dirname "$0")/../.env" ]]; then
  base64 < "$(dirname "$0")/../.env" | ssh "${ssh_args[@]}" "$remote" \
    'base64 -d > /tmp/runflow.env'
fi

ssh "${ssh_args[@]}" "$remote" bash -s <<'REMOTE'
set -euo pipefail
chmod 0600 /root/.ssh/id_ed25519
chmod 0600 /root/.config/rclone/rclone.conf

nix shell nixpkgs#git nixpkgs#openssh --command bash <<'BOOTSTRAP'
set -euo pipefail
ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null
test -d /workspace/styletts_studio_v2/.git || \
  git clone git@github.com:opentalk-org/StyleTTS2.1.git /workspace/styletts_studio_v2
cd /workspace/styletts_studio_v2
git fetch origin main
git switch --force-create main origin/main
BOOTSTRAP

mkdir -p /usr/local/nvidia/lib64
if [[ ! -e /usr/local/nvidia/lib64/libcuda.so.1 ]]; then
  driver_dir="$(dirname "$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)")"
  ln -s "$driver_dir/libcuda.so.1" \
    /usr/local/nvidia/lib64/libcuda.so.1
  unset driver_dir
fi

cd /workspace/styletts_studio_v2
nix shell nixpkgs#git nixpkgs#tmux --command nix develop \
  --command bash <<'DEVELOP'
set -euo pipefail
if [[ -f /tmp/runflow.env ]]; then
  mv /tmp/runflow.env .env
fi

# export NIX_LD="$(gcc -print-file-name=ld-linux-x86-64.so.2)"
# export NIX_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
uv sync --frozen
. .venv/bin/activate
cd src/frontend
npm ci --no-audit --no-fund
cd ../..

rclone copyto \
  r2train:training-data-eec/db_dump.dump \
  /workspace/db_dump.dump \
  --progress

for process in alembic backend frontend mlflow pg pgbouncer runner s3; do
  mkdir -p ".dnvr/runtime/$process"
  touch ".dnvr/runtime/$process/pid" ".dnvr/runtime/$process/launch.lock"
done

tmux has-session -t runflow-dnvr 2>/dev/null || \
  tmux new-session -d -s runflow-dnvr \
    'cd /workspace/styletts_studio_v2 && exec dnvr up'
tmux pipe-pane -t runflow-dnvr -o 'cat >> /tmp/runflow-dnvr.log'

echo "Waiting for PostgreSQL before restoring the R2 dump"
export PGHOST=127.0.0.1
export PGPORT=5432
export PGUSER=runflow
export PGPASSWORD=runflow
for _ in {1..120}; do
  psql --dbname=postgres --no-password --tuples-only \
    --command='SELECT 1' >/dev/null 2>&1 && break
  sleep 2
done
psql --dbname=postgres --no-password --tuples-only \
  --command='SELECT 1' >/dev/null
dropdb --if-exists runflow_restore
createdb --template=template0 runflow_restore
pg_restore \
  --dbname=runflow_restore \
  --exit-on-error \
  --jobs=16 \
  --no-owner \
  --no-privileges \
  /workspace/db_dump.dump
RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:5432/runflow_restore \
  alembic upgrade head

psql --dbname=postgres --set=ON_ERROR_STOP=1 <<'SQL'
ALTER DATABASE runflow WITH ALLOW_CONNECTIONS false;
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'runflow' AND pid <> pg_backend_pid();
ALTER DATABASE runflow RENAME TO runflow_empty;
ALTER DATABASE runflow_restore RENAME TO runflow;
ALTER DATABASE runflow WITH ALLOW_CONNECTIONS true;
DROP DATABASE runflow_empty WITH (FORCE);
SQL
rm /workspace/db_dump.dump

tmux send-keys -t runflow-dnvr q Enter
for _ in {1..30}; do
  tmux has-session -t runflow-dnvr 2>/dev/null || break
  sleep 1
done
tmux kill-session -t runflow-dnvr 2>/dev/null || true
tmux new-session -d -s runflow-dnvr \
  'cd /workspace/styletts_studio_v2 && exec dnvr up'
tmux pipe-pane -t runflow-dnvr -o 'cat >> /tmp/runflow-dnvr.log'

for port in 5432 6432 9000 8001 5173 7860; do
  echo "Waiting for port $port"
  port_ready=false
  for attempt in {1..120}; do
    if timeout 2 bash -c "</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
      port_ready=true
      break
    fi
    if ! tmux has-session -t runflow-dnvr 2>/dev/null; then
      cat /tmp/runflow-dnvr.log
      exit 1
    fi
    if (( attempt % 10 == 0 )); then
      dnvr ps
    fi
    sleep 2
  done
  if [[ "$port_ready" != true ]]; then
    dnvr ps
    echo "Port $port did not become ready" >&2
    exit 1
  fi
  echo "Port $port ready"
done

env \
  RUNFLOW_PGBOUNCER_DATABASE_URL=postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow \
  alembic upgrade head

python -c \
  'from urllib.request import urlopen; urlopen("http://127.0.0.1:8001/openapi.json").read()'
python -c \
  'from urllib.request import urlopen; urlopen("http://127.0.0.1:7860/health").read()'
python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
dnvr ps
echo "Ready to train. Attach with: tmux attach -t runflow-dnvr"
DEVELOP
REMOTE

echo "Vast instance: $instance_id"
echo "SSH: ssh -p $ssh_port root@$ssh_host"
echo "SSH with local port forwarding:"
printf 'ssh -i ~/.ssh/vastai \\\n'
printf '  -p %q \\\n' "$ssh_port"
printf '  -o ExitOnForwardFailure=yes \\\n'
for port in 50051 5432 5173 8001 8009 7860; do
  printf '  -L %s:127.0.0.1:%s \\\n' "$port" "$port"
done
printf '  root@%q\n' "$ssh_host"
