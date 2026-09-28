# Linux development shell cache

The `Cache Linux dev shells` workflow builds every shell in
`devShells.x86_64-linux` and `devShells.aarch64-linux` on native Linux runners.
Pushes to `main` and manual runs publish the environments and their dependencies
to Cachix. Pull requests build the same environments, with read-only cache access
and no Cachix write token. macOS is not built by this workflow.

## One-time maintainer setup

1. Create a **public** cache at <https://app.cachix.org>, or use an existing public
   cache. Create a write token scoped to that cache.
2. In this repository's **Settings → Secrets and variables → Actions**, set:
   - Repository variable `CACHIX_CACHE`: the cache name, without `.cachix.org`.
   - Repository secret `CACHIX_AUTH_TOKEN`: the cache's write token.
3. After merging, run **Actions → Cache Linux dev shells → Run workflow** on
   `main`, or let the next push populate the cache. Wait for your architecture's
   job to succeed before expecting cache hits.

Publishing runs fail early with a setup message if either setting is missing.
The first run still has to build anything absent from upstream caches. Subsequent
runs reuse cached outputs and upload newly built paths as they finish.

## Use the cache locally

Install the Cachix client, then run once, replacing `CACHE_NAME` with the
repository's `CACHIX_CACHE` value:

```sh
nix run nixpkgs#cachix -- use CACHE_NAME
```

`cachix use` installs the cache URL and its actual public signing key. Follow its
instructions if your Nix daemon requires system-level configuration. Public cache
downloads do not need the write token.

Then enter the shell as usual and accept the flake's upstream cache configuration
when prompted:

```sh
nix develop
```

The flake extends your configured substituters and trusted keys, so accepting its
configuration preserves the Cachix cache configured above.

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
