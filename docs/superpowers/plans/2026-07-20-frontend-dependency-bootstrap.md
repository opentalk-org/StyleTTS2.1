# Frontend Dependency Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make frontend dependencies install and resynchronize reliably when either development entry point runs from a fresh or stale checkout.

**Architecture:** A shared shell helper decides whether the locked frontend installation is current and runs `npm ci` when needed. The full-stack and frontend-only launchers call it before starting their services.

**Tech Stack:** Bash, npm lockfiles, Nix development shell

## Global Constraints

- Run project commands through `nix develop --command ...`.
- Use `npm ci`; never resolve dependencies without `package-lock.json`.
- Do not retain temporary tests in the repository.
- Keep each source file below 300 lines.

---

### Task 1: Shared frontend dependency synchronization

**Files:**
- Create: `nix/frontend-dependencies.sh`
- Modify: `nix/runflow-dev.sh`
- Modify: `nix/frontend-dev.sh`
- Test temporarily: `.tmp_tests/frontend-dependencies-test.sh`

**Interfaces:**
- Consumes: `sync_frontend_dependencies FRONTEND_DIRECTORY`
- Produces: a current `node_modules` installation and `node_modules/.runflow-npm-ci-stamp`

- [ ] **Step 1: Write a failing temporary shell test**

Create a fixture with copied package manifests and a fake `npm` executable that
records `ci` calls. Assert synchronization runs for absent and incomplete
installations, does not run for an unchanged complete installation, and reruns
when either manifest is newer than the stamp.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
nix develop --command bash .tmp_tests/frontend-dependencies-test.sh
```

Expected: failure because `nix/frontend-dependencies.sh` does not exist.

- [ ] **Step 3: Implement the helper and wire both launchers**

Create `sync_frontend_dependencies()` with direct manifest validation, a Vite
executable completeness check, manifest timestamp checks, `npm ci`, and stamp
creation after success. Source the helper from each launcher. In
`runflow-dev.sh`, call it before service initialization; in `frontend-dev.sh`,
replace the directory-only check.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
nix develop --command bash .tmp_tests/frontend-dependencies-test.sh
```

Expected: all synchronization cases pass.

- [ ] **Step 5: Verify source quality**

Run:

```bash
nix develop --command bash -n nix/frontend-dependencies.sh nix/frontend-dev.sh nix/runflow-dev.sh
nix develop --command npm --prefix src/frontend run build
wc -l nix/frontend-dependencies.sh nix/frontend-dev.sh nix/runflow-dev.sh
git diff --check
```

Expected: syntax and build pass, each file remains below 300 lines or the
already-oversized service script does not grow materially, and no whitespace
errors are reported.

- [ ] **Step 6: Remove the temporary test**

Delete `.tmp_tests/frontend-dependencies-test.sh` and remove `.tmp_tests` if it
is empty.

### Task 2: End-to-end startup verification

**Files:**
- No source changes

**Interfaces:**
- Consumes: `runflow-dev-session`, backend port `8001`, frontend port `5173`
- Produces: runtime evidence that the shared stack bootstraps and remains healthy

- [ ] **Step 1: Restart the shared session**

Run `nix develop --command runflow-dev-stop`, then launch
`nix develop --command runflow-dev-session` as the normal workspace user.

- [ ] **Step 2: Probe application health**

Run:

```bash
curl --fail http://127.0.0.1:8001/health
curl --fail --head http://127.0.0.1:5173/
```

Expected: backend returns `{"status":"ok"}` and frontend returns HTTP 200.

- [ ] **Step 3: Review final scope**

Run `git status --short` and `git diff --check`. Confirm only the helper,
launchers, spec, and plan remain changed and no dependency directories or
temporary files are tracked.
