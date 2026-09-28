# Linux development shell cache

The `Cache Linux dev shells` workflow builds every shell in
`devShells.x86_64-linux` and `devShells.aarch64-linux` on native Linux runners.
Pushes to `main` and manual runs publish the environments and their dependencies
to the public `opentalk` Cachix cache. Pull requests build the same environments, with read-only cache access
and no Cachix write token. macOS is not built by this workflow.

## One-time maintainer setup

1. At <https://app.cachix.org>, create a write token scoped to the `opentalk` cache.
2. In this repository's **Settings → Secrets and variables → Actions**, set:
   - Repository variable `CACHIX_CACHE`: `opentalk`.
   - Repository secret `CACHIX_AUTH_TOKEN`: the cache's write token.
3. After merging, run **Actions → Cache Linux dev shells → Run workflow** on
   `main`, or let the next push populate the cache. Wait for your architecture's
   job to succeed before expecting cache hits.

Publishing runs fail early with a setup message if either setting is missing.
The first run still has to build anything absent from upstream caches. Subsequent
runs reuse cached outputs and upload newly built paths as they finish.

## Use the cache locally

Enter the shell as usual and accept the flake's cache configuration when prompted.
The `opentalk` cache URL and verified public signing key are included in `flake.nix`:

```sh
nix develop
```

The flake extends your configured substituters and trusted keys. Public cache
downloads do not need a token. If your Nix daemon requires system-level cache
configuration, run the following and follow Cachix's instructions:

```sh
nix run nixpkgs#cachix -- use opentalk
```

To switch caches, change `CACHIX_CACHE` and update the URL and public signing key
in `flake.nix` to match.

To check that a warmed shell can be downloaded without local builds:

```sh
nix develop --accept-flake-config --max-jobs 0 -c true
```

This fails on a missing output unless a remote builder is configured. Use the
same commit and `flake.lock` that CI built; shell changes can require new outputs.
Nix still evaluates the flake and downloads dependencies on first use. Python
packages installed by `uv sync`, frontend `npm install` outputs, model weights,
and other files outside `/nix/store` are not part of this cache.

## How the workflow caches shells

`nix print-dev-env --profile ...` realizes the environment used by `nix develop`
without executing the shell hook or starting local services. Each resulting
profile is explicitly pushed with its dependency closure, including dependencies
that CI downloaded from another cache. Cachix's build hook also uploads completed
builds during the job, allowing later runs to reuse work if a build fails.

References: [Cachix shell caching](https://docs.cachix.org/pushing#pushing-shell-environment-1)
and [`nix print-dev-env`](https://nix.dev/manual/nix/latest/command-ref/new-cli/nix3-print-dev-env.html).
