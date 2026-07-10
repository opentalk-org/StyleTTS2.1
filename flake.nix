{
  description = "Runflow Studio single-image runtime with backend, runner, NATS JetStream, PgBouncer, and PostgreSQL";

  nixConfig = {
    substituters = [
      "https://cache.nixos.org"
      "https://cache.nixos-cuda.org"
    ];
    trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }:
    let
      imageSystem = "x86_64-linux";
      devSystems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      forAllDevSystems = nixpkgs.lib.genAttrs devSystems;

      importPkgs = system: import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      mkRustfs = pkgs: import ./nix/rustfs-package.nix { inherit pkgs; };

      mkPythonRuntime = pkgs:
        let
          cudaNvcc = pkgs.cudaPackages.cuda_nvcc;
          nvidiaDriverDirs = [
            "/usr/local/nvidia/lib"
            "/usr/local/nvidia/lib64"
          ];
          nvidiaDriverPath = pkgs.lib.concatStringsSep ":" nvidiaDriverDirs;
          runtimeLibs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.glibc
          ];
        in
        {
          python = pkgs.python312Full;
          inherit runtimeLibs;
          runtimeExecutableDeps = [
            pkgs.espeak-ng
            pkgs.ffmpeg-headless
            pkgs.gcc
            pkgs.cargo
            pkgs.openssl
            pkgs.patchelf
            pkgs.rustc
            cudaNvcc
          ];
          env = {
            CC = "${pkgs.gcc}/bin/gcc";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            PHONEMIZER_ESPEAK_LIBRARY = "${pkgs.espeak-ng}/lib/libespeak-ng.so";
            PYTHONUNBUFFERED = "1";
            TRITON_PTXAS_PATH = "${cudaNvcc}/bin/ptxas";
            TRITON_PTXAS_BLACKWELL_PATH = "${cudaNvcc}/bin/ptxas";
            TRITON_LIBCUDA_PATH = "/usr/local/nvidia/lib";
            # torchcodec (pulled by f5-tts / used by torchaudio 2.11 for decoding) dlopens
            # libavutil/libavcodec at import; expose ffmpeg-7's shared libs so core7 loads.
            # portaudio is added so fish-speech's pyaudio dep builds (Linux source-only) and loads.
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.ffmpeg-headless.lib}/lib:${pkgs.portaudio}/lib:${nvidiaDriverPath}";
            LIBRARY_PATH = "${pkgs.portaudio}/lib:${nvidiaDriverPath}";
            # The shellHook unsets NIX_CFLAGS_COMPILE, so pass portaudio's headers via CPATH
            # for the pyaudio C extension build.
            CPATH = "${pkgs.portaudio}/include";
          };
        };

      pythonEnvExports = pythonRuntime: ''
        export CC="${pythonRuntime.env.CC}"
        export SSL_CERT_FILE="${pythonRuntime.env.SSL_CERT_FILE}"
        export PHONEMIZER_ESPEAK_LIBRARY="${pythonRuntime.env.PHONEMIZER_ESPEAK_LIBRARY}"
        export PYTHONUNBUFFERED="${pythonRuntime.env.PYTHONUNBUFFERED}"
        export TRITON_PTXAS_PATH="${pythonRuntime.env.TRITON_PTXAS_PATH}"
        export TRITON_PTXAS_BLACKWELL_PATH="${pythonRuntime.env.TRITON_PTXAS_BLACKWELL_PATH}"
        export TRITON_LIBCUDA_PATH="${pythonRuntime.env.TRITON_LIBCUDA_PATH}"
        unset NIX_CFLAGS_COMPILE CFLAGS CXXFLAGS
        export LD_LIBRARY_PATH="${pythonRuntime.env.LD_LIBRARY_PATH}"
        export LIBRARY_PATH="${pythonRuntime.env.LIBRARY_PATH}"
        export CPATH="${pythonRuntime.env.CPATH}"
        export UV_PYTHON="${pythonRuntime.python}/bin/python${pythonRuntime.python.pythonVersion}"
        export UV_PYTHON_PREFERENCE="only-system"
        export UV_PYTHON_DOWNLOADS="never"
      '';

      mkFrontendStatic = pkgs: import ./nix/frontend-static.nix { inherit pkgs; };

      mkDevShell = system:
        let
          pkgs = importPkgs system;
          pythonRuntime = mkPythonRuntime pkgs;
          rustfs = mkRustfs pkgs;
          pythonTools = [ pkgs.uv pythonRuntime.python ] ++ pythonRuntime.runtimeExecutableDeps;
          runflowDev = pkgs.writeShellApplication {
            name = "runflow-dev";
            runtimeInputs = [
              pkgs.awscli2
              pkgs.bash
              pkgs.coreutils
              pkgs.gnugrep
              pkgs.nats-server
              pkgs.nodejs_22
              pkgs.pgbouncer
              pkgs.postgresql_16
              rustfs
            ] ++ pythonTools;
            text = (pythonEnvExports pythonRuntime) + builtins.readFile ./nix/runflow-dev.sh;
          };
          runflowDevSession = pkgs.writeShellApplication {
            name = "runflow-dev-session";
            runtimeInputs = [
              pkgs.bash
              pkgs.coreutils
              pkgs.gnugrep
              pkgs.zellij
              runflowDev
            ];
            text = ''
              set -euo pipefail

              session_name="runflow-dev"
              if zellij list-sessions --short --no-formatting | grep -Fxq "$session_name"; then
                echo "attaching to existing $session_name session"
              else
                echo "creating $session_name session and starting runflow-dev"
                zellij attach --create-background "$session_name"
                zellij --session "$session_name" run --name runflow-dev -- runflow-dev
              fi

              exec zellij attach --create "$session_name"
            '';
          };
          runflowDevStatus = pkgs.writeShellApplication {
            name = "runflow-dev-status";
            runtimeInputs = [
              pkgs.coreutils
              pkgs.gnugrep
              pkgs.zellij
            ];
            text = ''
              set -euo pipefail

              session_name="runflow-dev"
              if zellij list-sessions --short --no-formatting | grep -Fxq "$session_name"; then
                echo "$session_name session is running"
                zellij list-sessions --no-formatting | grep -F "$session_name"
              else
                echo "$session_name session is not running"
              fi
            '';
          };
          runflowDevStop = pkgs.writeShellApplication {
            name = "runflow-dev-stop";
            runtimeInputs = [
              pkgs.coreutils
              pkgs.gnugrep
              pkgs.zellij
            ];
            text = ''
              set -euo pipefail

              session_name="runflow-dev"
              if zellij list-sessions --short --no-formatting | grep -Fxq "$session_name"; then
                zellij kill-session "$session_name"
                echo "stopped $session_name session"
              else
                echo "$session_name session is not running"
              fi
            '';
          };
          frontendDev = pkgs.writeShellApplication {
            name = "runflow-frontend-dev";
            runtimeInputs = [
              pkgs.nodejs_22
            ];
            text = builtins.readFile ./nix/frontend-dev.sh;
          };
          runnerLaunch = pkgs.writeShellApplication {
            name = "runflow-runner-launch";
            runtimeInputs = pythonTools;
            text = (pythonEnvExports pythonRuntime) + builtins.readFile ./nix/runner-launch.sh;
          };
        in
        pkgs.mkShell {
          packages = [
            pkgs.awscli2
            runflowDev
            runflowDevSession
            runflowDevStatus
            runflowDevStop
            frontendDev
            runnerLaunch
            pkgs.nats-server
            pkgs.nodejs_22
            pkgs.pgbouncer
            pkgs.postgresql_16
            pkgs.uv
            pkgs.zellij
            pythonRuntime.python
            rustfs
          ] ++ pythonRuntime.runtimeExecutableDeps ++ pythonRuntime.runtimeLibs;

          shellHook = (pythonEnvExports pythonRuntime) + ''
            export PYTHONPATH="$PWD/src"

            if [ -e .venv/bin/activate ]; then
              . .venv/bin/activate
            else
              echo "no .venv yet - run: uv sync --frozen"
            fi
          '';
        };

      pkgs = importPkgs imageSystem;
      pythonRuntime = mkPythonRuntime pkgs;
      pythonTools = [ pkgs.uv pythonRuntime.python ] ++ pythonRuntime.runtimeExecutableDeps;
      # Runtime-only ML deps for the slim runner image: keeps the triton/inductor
      # JIT toolchain (gcc host compiler + nvcc ptxas) and audio libs, but drops the
      # build-only compilers (rustc/cargo/patchelf) and uv, since the venv is prebaked.
      runnerRuntimeDeps = [
        pkgs.espeak-ng
        pkgs.ffmpeg-headless
        pkgs.gcc
        pkgs.openssl
        pkgs.cudaPackages.cuda_nvcc
      ];
      frontendStatic = mkFrontendStatic pkgs;
      rustfs = mkRustfs pkgs;

      # Shared, non-secret defaults baked into BOTH images so the hub and the
      # remote runners agree on credentials, ports, and bucket names. Secrets
      # and topology (TAILSCALE_AUTHKEY, TAILSCALE_LOGIN_SERVER, prod passwords)
      # are injected at runtime via --env-file / Salad env, never baked here.
      commonEnv = [
        "POSTGRES_DB=runflow"
        "POSTGRES_USER=runflow"
        "POSTGRES_PASSWORD=runflow"
        "NATS_PORT=4222"
        "PGBOUNCER_PORT=6432"
        "RUSTFS_ACCESS_KEY=runflow"
        "RUSTFS_SECRET_KEY=runflow-secret"
        "RUSTFS_BUCKET=runflow"
        "AWS_ACCESS_KEY_ID=runflow"
        "AWS_SECRET_ACCESS_KEY=runflow-secret"
        "AWS_REGION=us-east-1"
      ];

      # Hub-only: local bind addresses/paths for the services it hosts, plus its
      # tailnet identity. Kernel/TUN tailscale (USERSPACE=0) so its 0.0.0.0
      # services are reachable by tailnet peers.
      hubEnv = [
        "PGDATA=/data/postgres"
        "PGHOST=/tmp/postgres"
        "PGPORT=5432"
        "NATS_DATA=/data/nats"
        "NATS_URL=nats://127.0.0.1:4222"
        "RUSTFS_DATA=/data/rustfs"
        "RUSTFS_VOLUMES=/data/rustfs"
        "RUSTFS_ADDRESS=0.0.0.0:9000"
        "RUSTFS_CONSOLE_ENABLE=true"
        "RUSTFS_CONSOLE_ADDRESS=0.0.0.0:9001"
        "AWS_ENDPOINT_URL=http://127.0.0.1:9000"
        "BACKEND_PORT=8000"
        "RUNNER_ID=runner-1"
        "TAILSCALE_HOSTNAME=runflow-hub"
        # Vast.ai managed containers have no /dev/net/tun, so the hub uses
        # userspace tailscale and exposes its services to the tailnet with
        # `tailscale serve --tcp` (raw TCP passthrough) on these ports.
        "TAILSCALE_USERSPACE=1"
        "TAILSCALE_SERVE_PORTS=6432 4222 9000"
      ];

      # Runner-only: the hub name it dials over the tailnet, plus userspace
      # tailscale (USERSPACE=1) since Salad containers have no TUN device. The
      # NATS/DB/S3 URLs are derived from RUNFLOW_HUB_HOST in runner-entrypoint.sh.
      # NOTE: RUNNER_ID is intentionally NOT baked — it must be unique per Salad
      # replica (it keys the runner's DB heartbeat row and NATS work routing).
      # runner-entrypoint.sh derives it from $SALAD_MACHINE_ID at start.
      runnerEnv = [
        "RUNFLOW_HUB_HOST=runflow-hub"
        "TAILSCALE_USERSPACE=1"
        # Root has no home in the minimal image; give torch/HF/etc a writable
        # cache on the /data volume (getpass.getuser needs fakeNss, added below).
        "HOME=/data"
        "XDG_CACHE_HOME=/data/cache"
        "HF_HOME=/data/huggingface"
        "TORCHINDUCTOR_CACHE_DIR=/data/torchinductor"
      ];

      # Fully-baked Python environment (torch/vllm/flashinfer/... ~13GB) built by
      # Nix so the runner image is self-contained and needs no runtime `uv sync`.
      # A fixed-output derivation is the one build type allowed network access in
      # the sandbox (identically on GitHub CI), so uv can fetch the pinned wheels,
      # custom indexes (pytorch-cu128, the vLLM cu129 wheel), and git sources.
      # UV_PROJECT_ENVIRONMENT=$out puts the venv at its final store path, so the
      # venv's script shebangs are correct at runtime (no relocation needed).
      runnerVenv = pkgs.stdenv.mkDerivation {
        name = "runflow-runner-venv";
        # Only the resolver inputs, so venv rebuilds track uv.lock, not app code.
        src = pkgs.runCommand "runflow-venv-src" { } ''
          mkdir -p "$out"
          cp ${./pyproject.toml} "$out/pyproject.toml"
          cp ${./uv.lock} "$out/uv.lock"
        '';
        nativeBuildInputs = [
          pkgs.uv
          pythonRuntime.python
          pkgs.cacert
          pkgs.git
        ] ++ pythonRuntime.runtimeExecutableDeps ++ pythonRuntime.runtimeLibs;

        # Impure build: uv needs network for the pinned wheels, the pytorch-cu128
        # index, the vLLM cu129 wheel, and git sources. A fixed-output hash is not
        # usable here because source-built extensions (monotonic-align, fish-speech)
        # embed their build path into the compiled .so, so the output is not
        # byte-reproducible. __noChroot keeps it a normal (input-addressed, cacheable)
        # derivation with network access; the builder must allow it (sandbox=relaxed,
        # set in the flake's nixConfig and the CI workflow).
        __noChroot = true;

        dontConfigure = true; # no autotools/cmake project here; we drive uv directly
        dontFixup = true; # keep wheel .so RPATHs intact

        buildPhase = (pythonEnvExports pythonRuntime) + ''
          export HOME="$TMPDIR"
          export UV_CACHE_DIR="$TMPDIR/uvcache"
          export UV_PYTHON="${pythonRuntime.python}/bin/python${pythonRuntime.python.pythonVersion}"
          export UV_PYTHON_DOWNLOADS=never
          export UV_NO_BYTECODE=1
          export UV_LINK_MODE=copy
          export UV_PROJECT_ENVIRONMENT="$out"
          uv sync --frozen --no-install-project --no-editable
        '';

        installPhase = ''
          # Drop non-deterministic bytecode so the output hash is stable.
          find "$out" -depth -type d -name '__pycache__' -exec rm -rf {} + || true
          find "$out" -type f -name '*.pyc' -delete || true
        '';
      };

      runflowBackend = pkgs.writeShellApplication {
        name = "runflow-backend";
        runtimeInputs = pythonTools;
        text = (pythonEnvExports pythonRuntime) + ''
          cd ${./.}
          export PYTHONPATH="${./src}"
          # PYTHONPATH points at an isolated src/ store path, so the Alembic dir cannot
          # be found relative to the source tree; point at it explicitly.
          export RUNFLOW_ALEMBIC_DIR="${./migrations}"
          export RUNFLOW_UI_STATIC_DIR="${frontendStatic}"
          exec uv run --frozen uvicorn backend.api:app --host 0.0.0.0 --port "''${BACKEND_PORT:-8000}"
        '';
      };

      runflowRunner = pkgs.writeShellApplication {
        name = "runflow-runner";
        runtimeInputs = pythonTools;
        text = (pythonEnvExports pythonRuntime) + ''
          cd ${./.}
          export PYTHONPATH="${./src}"
          exec uv run --frozen python -m runner.cli \
            --runner-id "''${RUNNER_ID:-runner-1}" \
            --nats-url "''${NATS_URL:-nats://127.0.0.1:4222}"
        '';
      };
      runnerLaunch = pkgs.writeShellApplication {
        name = "runflow-runner-launch";
        runtimeInputs = pythonTools;
        text = (pythonEnvExports pythonRuntime) + builtins.readFile ./nix/runner-launch.sh;
      };

      # runner-launch backed by the prebaked venv (no uv, no runtime sync). Puts
      # the venv on PATH so `python` in runner-launch.sh is the venv interpreter
      # with torch/vllm/... already installed.
      runnerLaunchVenv = pkgs.writeShellApplication {
        name = "runflow-runner-launch";
        runtimeInputs = [ pkgs.bash pkgs.coreutils ]
          ++ runnerRuntimeDeps ++ pythonRuntime.runtimeLibs;
        text = (pythonEnvExports pythonRuntime) + ''
          export VIRTUAL_ENV="${runnerVenv}"
          export PATH="${runnerVenv}/bin:$PATH"
          export PYTHONPATH="${./src}"
        '' + builtins.readFile ./nix/runner-launch.sh;
      };

      runflowAim = pkgs.writeShellApplication {
        name = "runflow-aim";
        runtimeInputs = pythonTools;
        text = (pythonEnvExports pythonRuntime) + ''
          cd ${./.}
          export PYTHONPATH="${./src}"
          AIM_REPO="''${AIM_REPO:-/data/aim}"
          AIM_HOST="''${AIM_HOST:-0.0.0.0}"
          AIM_PORT="''${AIM_PORT:-43800}"
          mkdir -p "$AIM_REPO"
          if [ ! -d "$AIM_REPO/.aim" ]; then
            uv run --frozen aim init --repo "$AIM_REPO" || echo "aim init failed; continuing without Aim UI"
          fi
          exec uv run --frozen aim up --repo "$AIM_REPO" --host "$AIM_HOST" --port "$AIM_PORT"
        '';
      };

      # Shared Headscale/Tailscale bring-up, invoked (not sourced) by both
      # entrypoints. Backgrounds tailscaled (kernel or userspace per
      # TAILSCALE_USERSPACE) and runs `tailscale up`; the daemon survives this
      # command's exit so the parent entrypoint keeps its tailnet + SOCKS proxy.
      tailscaleUp = pkgs.writeShellApplication {
        name = "tailscale-up";
        runtimeInputs = [
          pkgs.bash
          pkgs.coreutils
          pkgs.gnugrep
          pkgs.iproute2
          pkgs.tailscale
        ];
        text = builtins.readFile ./nix/tailscale-up.sh;
      };

      entrypoint = pkgs.writeShellApplication {
        name = "runflow-entrypoint";
        runtimeInputs = [
          pkgs.awscli2
          pkgs.bash
          pkgs.coreutils
          pkgs.gnugrep
          pkgs.gnused
          pkgs.nats-server
          pkgs.pgbouncer
          pkgs.postgresql_16
          pkgs.tailscale
          tailscaleUp
          rustfs
          runflowBackend
          runflowRunner
          runnerLaunch
          runflowAim
        ] ++ pythonTools;
        text = builtins.readFile ./nix/entrypoint.sh;
      };

      # Slim runner entrypoint: userspace tailscale + proxychains-ng so the
      # runner's raw-TCP Postgres/NATS clients reach the hub over the tailnet.
      runnerEntrypoint = pkgs.writeShellApplication {
        name = "runflow-runner-entrypoint";
        runtimeInputs = [
          pkgs.bash
          pkgs.coreutils
          pkgs.gnugrep
          pkgs.iproute2
          pkgs.proxychains-ng
          pkgs.tailscale
          tailscaleUp
          runnerLaunchVenv
        ];
        text = builtins.readFile ./nix/runner-entrypoint.sh;
      };
    in
    {
      devShells = forAllDevSystems (system: {
        default = mkDevShell system;
      });

      packages = forAllDevSystems (system:
        let
          pkgsForSystem = importPkgs system;
          rustfsForSystem = mkRustfs pkgsForSystem;
        in
        {
          frontend-static = mkFrontendStatic pkgsForSystem;
          rustfs = rustfsForSystem;
          default = self.packages.${system}.frontend-static;
        }
      ) // {
        ${imageSystem} = rec {
          frontend-static = frontendStatic;
          inherit rustfs;
          runner-venv = runnerVenv;

          image = pkgs.dockerTools.buildLayeredImage {
            name = "runflow-studio-single-image";
            tag = "latest";

            # Minimal nix images have no /tmp; many things (tailscaled logs,
            # PostgreSQL, Python tempfile) need it.
            extraCommands = "mkdir -p tmp && chmod 1777 tmp";

            contents = [
              pkgs.bash
              pkgs.cacert
              pkgs.coreutils
              pkgs.dockerTools.fakeNss
              pkgs.nats-server
              pkgs.pgbouncer
              pkgs.postgresql_16
              pkgs.tailscale
              rustfs
              frontendStatic
              runflowBackend
              runflowRunner
              runflowAim
              entrypoint
            ] ++ pythonRuntime.runtimeExecutableDeps ++ pythonRuntime.runtimeLibs;

            config = {
              Cmd = [ "${entrypoint}/bin/runflow-entrypoint" ];

              ExposedPorts = {
                "8000/tcp" = {};
                "9000/tcp" = {};
                "9001/tcp" = {};
                "4222/tcp" = {};
                "6432/tcp" = {};
                "43800/tcp" = {};
              };

              Env = commonEnv ++ hubEnv;

              Volumes = {
                "/data" = {};
              };
            };
          };

          # Slim runner-only image for remote GPU workers (Salad). Drops every
          # server binary (postgres/pgbouncer/nats/rustfs/backend/frontend/aim);
          # keeps the ML runtime + tailscale + proxychains. Reaches the hub's
          # DB/NATS/S3 as a client over the tailnet.
          runner-image = pkgs.dockerTools.buildLayeredImage {
            name = "runflow-studio-runner";
            tag = "latest";

            # Minimal nix images have no /tmp; tailscaled, proxychains, and
            # Python tempfile all need it.
            extraCommands = "mkdir -p tmp && chmod 1777 tmp";

            contents = [
              pkgs.bash
              pkgs.cacert
              pkgs.coreutils
              pkgs.dockerTools.fakeNss
              pkgs.gnugrep
              pkgs.iproute2
              pkgs.proxychains-ng
              pkgs.tailscale
              runnerVenv
              pythonRuntime.python
              runnerLaunchVenv
              runnerEntrypoint
            ] ++ runnerRuntimeDeps ++ pythonRuntime.runtimeLibs;

            config = {
              Cmd = [ "${runnerEntrypoint}/bin/runflow-runner-entrypoint" ];

              Env = commonEnv ++ runnerEnv;

              Volumes = {
                "/data" = {};
              };
            };
          };

          default = image;
        };
      };
      defaultPackage.${imageSystem} = self.packages.${imageSystem}.image;
    };
}
