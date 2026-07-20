# Rclone S3 Dev Service Design

## Goal

Extend the shared `nix develop` development stack with an authenticated S3-compatible endpoint at `http://127.0.0.1:8002`. The endpoint exposes `/home/storagebucket` on the SSH host alias `hetzner-storagebox` through rclone's SFTP backend. Rename the repository's full `RUSTFS_*` environment-variable family to the storage-neutral `BUCKET_*` family.

## Architecture

`flake.nix` supplies `rclone` and OpenSSH to the `runflow-dev` runtime. `nix/runflow-dev.sh` owns the service lifecycle alongside backend, frontend, MLflow, runner, PostgreSQL, PgBouncer, and RustFS.

The service runs the equivalent of:

```bash
rclone serve s3 \
  --addr 127.0.0.1:8002 \
  --auth-key "$BUCKET_ACCESS_KEY,$BUCKET_SECRET_KEY" \
  --sftp-ssh "ssh hetzner-storagebox" \
  :sftp:/home/storagebucket
```

The existing credential values move to `BUCKET_ACCESS_KEY` and `BUCKET_SECRET_KEY` in `.env` and are the only S3 credentials. Both RustFS and rclone consume them. SSH authentication and host settings remain owned by the normal development user's OpenSSH configuration.

## Configuration

The script declares explicit environment settings:

- `RCLONE_S3_HOST`, defaulting to `127.0.0.1`.
- `RCLONE_S3_PORT`, defaulting to `8002`.
- `RCLONE_S3_PATH`, defaulting to `/home/storagebucket`.
- `RCLONE_S3_SSH`, defaulting to `ssh hetzner-storagebox`.

Every current `RUSTFS_*` setting is renamed without a compatibility alias:

- `RUSTFS_DATA` becomes `BUCKET_DATA`.
- `RUSTFS_VOLUMES` becomes `BUCKET_VOLUMES`.
- `RUSTFS_ADDRESS` becomes `BUCKET_ADDRESS`.
- `RUSTFS_CONSOLE_ENABLE` becomes `BUCKET_CONSOLE_ENABLE`.
- `RUSTFS_CONSOLE_ADDRESS` becomes `BUCKET_CONSOLE_ADDRESS`.
- `RUSTFS_ACCESS_KEY` becomes `BUCKET_ACCESS_KEY`.
- `RUSTFS_SECRET_KEY` becomes `BUCKET_SECRET_KEY`.
- `RUSTFS_BUCKET` becomes `BUCKET_NAME`.

The rename applies to `flake.nix`, local development, hub and runner entrypoints, and the deployment environment example. `RUNFLOW_S3_*`, `AWS_*`, and `MLFLOW_*` interfaces keep their names and derive their defaults from `BUCKET_*`. No legacy fallback is retained because the project is greenfield.

These settings allow local overrides without adding credentials or host-specific SSH material to the repository. The access key and secret are not duplicated into rclone-specific variables.

## Lifecycle and Failure Behavior

The rclone process uses the existing `supervise` helper so it restarts after an unexpected exit. Startup frees port 8002 through the existing `free_port` helper before launching the service. The process identifier participates in the existing shutdown trap so `runflow-dev-stop` terminates it with the rest of the shared stack.

The dev stack does not require an initial remote listing before starting backend and frontend. A temporarily unavailable SSH host therefore causes S3 requests to fail visibly while the rest of the development services remain available.

## Verification

Validation uses Nix commands only:

1. Evaluate the dev shell and confirm `rclone` is available.
2. Run a temporary shell-level assertion against the assembled rclone command and lifecycle wiring before implementation, then remove the temporary test.
3. Restart the shared `runflow-dev` Zellij session.
4. Confirm the session and rclone process run as the normal `user` account.
5. Confirm no `RUSTFS_*` references remain in repository configuration or scripts.
6. Use the AWS CLI with `BUCKET_ACCESS_KEY` and `BUCKET_SECRET_KEY` plus `--endpoint-url http://127.0.0.1:8002` to list buckets, proving S3 authentication and SSH-backed access work together.

No committed test file is retained because the repository explicitly prefers throwaway tests unless persistent tests are requested.
