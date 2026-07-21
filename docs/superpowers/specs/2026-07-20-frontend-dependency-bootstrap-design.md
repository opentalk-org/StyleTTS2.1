# Frontend Dependency Bootstrap Design

## Goal

Starting the development stack or standalone frontend from a fresh clone must
install the exact frontend dependencies from `package-lock.json`. Startup must
also repair an incomplete `node_modules` directory and resync dependencies after
either frontend package manifest changes.

## Design

A focused shell helper under `nix/` owns frontend dependency synchronization.
It receives the frontend directory, checks a stamp stored inside `node_modules`,
and runs `npm ci` when the installation is absent, incomplete, unstamped, or
older than `package.json` or `package-lock.json`. A successful install updates
the stamp. A failed install leaves no current stamp and fails startup visibly.

Both `nix/runflow-dev.sh` and `nix/frontend-dev.sh` source and call the helper
before launching long-running processes. The full stack performs this check
before PostgreSQL or any application service starts, avoiding a partially
started stack when frontend bootstrapping fails. The standalone frontend uses
the identical behavior so the two entry points cannot drift.

The helper checks the Vite executable as the concrete completeness invariant
because Vite is the frontend entry point. The timestamp comparison covers
dependency changes without reinstalling unchanged packages on every launch.

## Failure Behavior

Missing `package.json` or `package-lock.json`, an unavailable `npm`, and a failed
`npm ci` are hard errors. No fallback install mode or unlocked dependency
resolution is used.

## Verification

Temporary shell checks exercise an absent install, an incomplete install, an
unchanged stamped install, and a package-manifest timestamp newer than the
stamp. Repository verification then runs the frontend production build and the
shared development stack, requiring successful HTTP responses from the backend
health endpoint and frontend root.
