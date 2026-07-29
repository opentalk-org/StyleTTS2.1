#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: ./nix/run-venv.sh <command> [arguments...]" >&2
  exit 64
fi

script_path="$(realpath "${BASH_SOURCE[0]}")"
project_root="$(dirname "$(dirname "$script_path")")"
cd "$project_root"

if [ "${RUNFLOW_VENV_ACTIVE:-0}" != "1" ]; then
  exec nix develop --command env RUNFLOW_VENV_ACTIVE=1 "$script_path" "$@"
fi

export POSTGRES_DB="${POSTGRES_DB:-runflow}"
export POSTGRES_USER="${POSTGRES_USER:-runflow}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-runflow}"
export PGBOUNCER_PORT="${PGBOUNCER_PORT:-6432}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:7860}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-runflow}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-runflow-secret}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://127.0.0.1:9000}"
if [ -z "${RUNFLOW_PGBOUNCER_DATABASE_URL:-}" ]; then
  printf -v RUNFLOW_PGBOUNCER_DATABASE_URL \
    'postgresql+psycopg://%s:%s@127.0.0.1:%s/%s' \
    "$POSTGRES_USER" \
    "$POSTGRES_PASSWORD" \
    "$PGBOUNCER_PORT" \
    "$POSTGRES_DB"
fi
export RUNFLOW_PGBOUNCER_DATABASE_URL
export PYTHONPATH="$project_root/src"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

if [ "$(id -u)" -ne 0 ]; then
  exec "$@"
fi

target_user="${RUNFLOW_DEV_USER:-user}"
if ! id "$target_user" >/dev/null 2>&1; then
  echo "run-venv target user does not exist: $target_user" >&2
  exit 1
fi

target_home="$(getent passwd "$target_user" | cut -d: -f6)"
exec runuser -u "$target_user" -- env \
  HOME="$target_home" \
  USER="$target_user" \
  LOGNAME="$target_user" \
  TMPDIR=/tmp \
  XDG_CACHE_HOME="$target_home/.cache" \
  XDG_DATA_HOME="$target_home/.local/share" \
  XDG_STATE_HOME="$target_home/.local/state" \
  "$@"
