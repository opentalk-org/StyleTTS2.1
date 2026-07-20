# Rclone S3 Dev Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a supervised, SSH-backed rclone S3 endpoint on `127.0.0.1:8002` and replace the full `RUSTFS_*` environment contract with `BUCKET_*`.

**Architecture:** The Nix dev shell provides rclone and OpenSSH. The shared `runflow-dev` process loads `.env`, starts rclone against an on-the-fly SFTP backend using `ssh hetzner-storagebox`, supervises it independently, and shuts it down with the stack. RustFS, rclone, backend storage, MLflow, hub images, and runner images consume one storage-neutral `BUCKET_*` contract.

**Tech Stack:** Nix flakes, Bash, rclone S3 server, rclone SFTP backend, OpenSSH, AWS CLI, Zellij

## Global Constraints

- Run all project commands through `nix develop --command ...`.
- Use the single shared `runflow-dev` Zellij session; never start a second stack.
- Run service processes as the normal `user` account.
- Rename every `RUSTFS_*` variable without compatibility aliases.
- Do not retain temporary tests in the repository.
- Do not commit `.env` or expose its values in command output.

---

### Task 1: Storage-neutral environment contract

**Files:**
- Modify: `flake.nix`
- Modify: `nix/runflow-dev.sh`
- Modify: `nix/entrypoint.sh`
- Modify: `nix/runner-entrypoint.sh`
- Modify: `deploy/hub.env.example`
- Modify locally, do not commit: `.env`
**Interfaces:**
- Consumes: the current `RUSTFS_DATA`, `RUSTFS_VOLUMES`, `RUSTFS_ADDRESS`, `RUSTFS_CONSOLE_ENABLE`, `RUSTFS_CONSOLE_ADDRESS`, `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`, and `RUSTFS_BUCKET` settings.
- Produces: `BUCKET_DATA`, `BUCKET_VOLUMES`, `BUCKET_ADDRESS`, `BUCKET_CONSOLE_ENABLE`, `BUCKET_CONSOLE_ADDRESS`, `BUCKET_ACCESS_KEY`, `BUCKET_SECRET_KEY`, and `BUCKET_NAME` with the same values and consumers.

- [ ] **Step 1: Write the temporary failing contract test** — Create `/tmp/check-bucket-env.sh` with `apply_patch`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/styletts_studio_v2
tracked=(
  "$repo/flake.nix"
  "$repo/nix/runflow-dev.sh"
  "$repo/nix/entrypoint.sh"
  "$repo/nix/runner-entrypoint.sh"
  "$repo/deploy/hub.env.example"
)

! rg -n 'RUSTFS_[A-Z0-9_]+' "${tracked[@]}"
rg -q 'BUCKET_ACCESS_KEY' "$repo/flake.nix"
rg -q 'BUCKET_SECRET_KEY' "$repo/nix/runflow-dev.sh"
rg -q 'BUCKET_NAME' "$repo/nix/entrypoint.sh"
rg -q 'BUCKET_NAME' "$repo/nix/runner-entrypoint.sh"
```

- [ ] **Step 2: Run the test and verify the intended failure** — Run:

```bash
nix develop --command bash /tmp/check-bucket-env.sh
```

Expected: FAIL because tracked files still contain `RUSTFS_*` and do not yet contain the full `BUCKET_*` contract.

- [ ] **Step 3: Rename the environment variables and load `.env` in the dev shell** — In `flake.nix`, add this at the start of `shellHook`, before project defaults are consumed:

```bash
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
```

Rename the image variables exactly:

```nix
"BUCKET_ACCESS_KEY=runflow"
"BUCKET_SECRET_KEY=runflow-secret"
"BUCKET_NAME=runflow"
"BUCKET_DATA=/data/rustfs"
"BUCKET_VOLUMES=/data/rustfs"
"BUCKET_ADDRESS=0.0.0.0:9000"
"BUCKET_CONSOLE_ENABLE=true"
"BUCKET_CONSOLE_ADDRESS=0.0.0.0:9001"
```

In all three shell entrypoints, replace the old declarations and downstream defaults with:

```bash
export BUCKET_DATA="${BUCKET_DATA:-/data/rustfs}"
export BUCKET_VOLUMES="${BUCKET_VOLUMES:-$BUCKET_DATA}"
export BUCKET_ADDRESS="${BUCKET_ADDRESS:-0.0.0.0:9000}"
export BUCKET_CONSOLE_ENABLE="${BUCKET_CONSOLE_ENABLE:-true}"
export BUCKET_CONSOLE_ADDRESS="${BUCKET_CONSOLE_ADDRESS:-0.0.0.0:9001}"
export BUCKET_ACCESS_KEY="${BUCKET_ACCESS_KEY:-runflow}"
export BUCKET_SECRET_KEY="${BUCKET_SECRET_KEY:-runflow-secret}"
export BUCKET_NAME="${BUCKET_NAME:-runflow}"
export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET_KEY"
export RUNFLOW_S3_BUCKET="${RUNFLOW_S3_BUCKET:-$BUCKET_NAME}"
export RUNFLOW_S3_ACCESS_KEY_ID="${RUNFLOW_S3_ACCESS_KEY_ID:-$BUCKET_ACCESS_KEY}"
export RUNFLOW_S3_SECRET_ACCESS_KEY="${RUNFLOW_S3_SECRET_ACCESS_KEY:-$BUCKET_SECRET_KEY}"
```

Use `.data/rustfs` and `127.0.0.1:9000` for the corresponding local-dev defaults already present in `nix/runflow-dev.sh`. Rename all path creation, RustFS bucket creation, and MLflow artifact references to the new variables. Change the deployment example to `# BUCKET_SECRET_KEY=runflow-secret`.

Add these untracked local entries to `.env` without printing their values:

```dotenv
BUCKET_ACCESS_KEY=runflow
BUCKET_SECRET_KEY=runflow-secret
```

- [ ] **Step 4: Run the contract test and Nix evaluation** — Run:

```bash
nix develop --command bash /tmp/check-bucket-env.sh
nix flake check --no-build
```

Expected: the contract test exits 0; flake evaluation succeeds.

- [ ] **Step 5: Remove the temporary test and commit the tracked rename** — Remove `/tmp/check-bucket-env.sh`, then run:

```bash
git add flake.nix nix/runflow-dev.sh nix/entrypoint.sh nix/runner-entrypoint.sh deploy/hub.env.example
git commit -m "refactor: use storage-neutral bucket environment"
```

Expected: commit succeeds; `.env` remains ignored and unstaged.

### Task 2: Supervised rclone S3 service

**Files:**
- Modify: `flake.nix`
- Modify: `nix/runflow-dev.sh`
**Interfaces:**
- Consumes: `BUCKET_ACCESS_KEY`, `BUCKET_SECRET_KEY`, the existing `free_port`, `kill_tree`, and `supervise` shell functions, and the user's OpenSSH host alias `hetzner-storagebox`.
- Produces: an authenticated S3 endpoint at `http://$RCLONE_S3_HOST:$RCLONE_S3_PORT`, defaulting to `http://127.0.0.1:8002`.

- [ ] **Step 1: Write the temporary failing lifecycle test** — Create `/tmp/check-rclone-dev-service.sh` with `apply_patch`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/styletts_studio_v2
flake="$repo/flake.nix"
script="$repo/nix/runflow-dev.sh"

rg -q 'pkgs.rclone' "$flake"
rg -q 'pkgs.openssh' "$flake"
rg -q 'RCLONE_S3_PORT="\${RCLONE_S3_PORT:-8002}"' "$script"
rg -q 'RCLONE_S3_PATH="\${RCLONE_S3_PATH:-/home/storagebucket}"' "$script"
rg -q 'RCLONE_S3_SSH="\${RCLONE_S3_SSH:-ssh hetzner-storagebox}"' "$script"
rg -q 'supervise rclone-s3 --port "\$RCLONE_S3_PORT" rclone serve s3' "$script"
rg -q '\[ -z "\$pid_rclone_s3" \] || kill "\$pid_rclone_s3"' "$script"
rg -q '\[ -z "\$pid_rclone_s3" \] || wait_pids+=' "$script"
```

- [ ] **Step 2: Run the lifecycle test and verify the intended failure** — Run:

```bash
nix develop --command bash /tmp/check-rclone-dev-service.sh
```

Expected: FAIL at the first missing rclone integration assertion.

- [ ] **Step 3: Add rclone and OpenSSH to the Nix dev runtime** — Add `pkgs.rclone` and `pkgs.openssh` to `runflowDev.runtimeInputs` and the dev shell `packages` list in `flake.nix`:

```nix
pkgs.openssh
pkgs.rclone
```

- [ ] **Step 4: Add configuration and lifecycle wiring** — Add local defaults near the other service settings in `nix/runflow-dev.sh`:

```bash
export RCLONE_S3_HOST="${RCLONE_S3_HOST:-127.0.0.1}"
export RCLONE_S3_PORT="${RCLONE_S3_PORT:-8002}"
export RCLONE_S3_PATH="${RCLONE_S3_PATH:-/home/storagebucket}"
export RCLONE_S3_SSH="${RCLONE_S3_SSH:-ssh hetzner-storagebox}"
```

Declare `pid_rclone_s3=""`. After `supervise` is defined and before the backend starts, launch:

```bash
echo "Starting rclone S3 at http://$RCLONE_S3_HOST:$RCLONE_S3_PORT"
free_port "$RCLONE_S3_PORT" rclone-s3
supervise rclone-s3 --port "$RCLONE_S3_PORT" rclone serve s3 \
  --addr "$RCLONE_S3_HOST:$RCLONE_S3_PORT" \
  --auth-key "$BUCKET_ACCESS_KEY,$BUCKET_SECRET_KEY" \
  --sftp-ssh "$RCLONE_S3_SSH" \
  ":sftp:$RCLONE_S3_PATH" &
pid_rclone_s3=$!
```

Add this before the one-second shutdown delay:

```bash
[ -z "$pid_rclone_s3" ] || kill "$pid_rclone_s3" 2>/dev/null || true
```

Add this to `wait_pids`:

```bash
[ -z "$pid_rclone_s3" ] || wait_pids+=("$pid_rclone_s3")
```

- [ ] **Step 5: Run the lifecycle test and Nix checks** — Run:

```bash
nix develop --command bash /tmp/check-rclone-dev-service.sh
nix develop --command rclone version
nix flake check --no-build
```

Expected: both temporary assertions and flake evaluation pass; `rclone version` reports the Nix-provided binary.

- [ ] **Step 6: Remove the temporary test and commit the service** — Remove `/tmp/check-rclone-dev-service.sh`, then run:

```bash
git add flake.nix nix/runflow-dev.sh
git commit -m "feat: serve ssh storage through s3 in dev"
```

### Task 3: Live shared-stack verification

**Files:**
- No tracked file changes.
**Interfaces:**
- Consumes: the shared `runflow-dev` session, `.env`, the user's SSH configuration, and the endpoint from Task 2.
- Produces: runtime evidence that backend, frontend, runner, RustFS, and rclone are alive under the normal user.

- [ ] **Step 1: Restart only the shared stack** — Run:

```bash
env -u ZELLIJ_CONFIG_DIR nix develop --command runflow-dev-stop
env -u ZELLIJ_CONFIG_DIR nix develop --command runflow-dev-session
```

Detach from Zellij after startup. Expected output includes `Starting rclone S3 at http://127.0.0.1:8002`.

- [ ] **Step 2: Verify session and processes** — Run:

```bash
env -u ZELLIJ_CONFIG_DIR nix develop --command runflow-dev-status
ps -eo user,comm,args | rg 'rclone serve s3|uvicorn backend.api|vite --host|runner.cli --runner-id'
```

Expected: one shared session; all listed processes are owned by `user`; rclone includes port 8002, the SSH alias, and `/home/storagebucket`.

- [ ] **Step 3: Verify local application health** — Run:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsSI http://127.0.0.1:5173/ | head -n 1
curl -fsS http://127.0.0.1:7860/health
```

Expected: backend returns `{"status":"ok"}`, frontend returns `HTTP/1.1 200 OK`, and MLflow returns `OK`.

- [ ] **Step 4: Verify signed access to the SSH-backed S3 endpoint** — Run without printing credentials:

```bash
nix develop --command bash -c '
  export AWS_ACCESS_KEY_ID="$BUCKET_ACCESS_KEY"
  export AWS_SECRET_ACCESS_KEY="$BUCKET_SECRET_KEY"
  aws --endpoint-url http://127.0.0.1:8002 s3api list-buckets
'
```

Expected: exit 0 and JSON containing the directories exposed as buckets beneath `/home/storagebucket`. If SSH connectivity fails, inspect the shared Zellij output and correct the user's SSH configuration rather than adding repository credentials or fallback behavior.

- [ ] **Step 5: Run final repository checks** — Run:

```bash
! rg -n 'RUSTFS_[A-Z0-9_]+' flake.nix nix deploy/hub.env.example
git diff --check
git status --short
```

Expected: no legacy environment references, no whitespace errors, and only intentional changes or commits.
