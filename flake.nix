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
          python = pkgs.python312;
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
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${nvidiaDriverPath}";
            LIBRARY_PATH = nvidiaDriverPath;
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
            frontendDev
            runnerLaunch
            pkgs.nats-server
            pkgs.nodejs_22
            pkgs.pgbouncer
            pkgs.postgresql_16
            pkgs.uv
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
      frontendStatic = mkFrontendStatic pkgs;
      rustfs = mkRustfs pkgs;

      runflowBackend = pkgs.writeShellApplication {
        name = "runflow-backend";
        runtimeInputs = pythonTools;
        text = (pythonEnvExports pythonRuntime) + ''
          cd ${./.}
          export PYTHONPATH="${./src}"
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
          rustfs
          runflowBackend
          runflowRunner
          runnerLaunch
          runflowAim
        ] ++ pythonTools;
        text = builtins.readFile ./nix/entrypoint.sh;
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

          image = pkgs.dockerTools.buildLayeredImage {
            name = "runflow-studio-single-image";
            tag = "latest";

            contents = [
              pkgs.bash
              pkgs.cacert
              pkgs.coreutils
              pkgs.nats-server
              pkgs.pgbouncer
              pkgs.postgresql_16
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

              Env = [
                "PGDATA=/data/postgres"
                "PGHOST=/tmp/postgres"
                "PGPORT=5432"
                "PGBOUNCER_PORT=6432"
                "POSTGRES_DB=runflow"
                "POSTGRES_USER=runflow"
                "POSTGRES_PASSWORD=runflow"
                "NATS_DATA=/data/nats"
                "NATS_URL=nats://127.0.0.1:4222"
                "RUSTFS_DATA=/data/rustfs"
                "RUSTFS_VOLUMES=/data/rustfs"
                "RUSTFS_ADDRESS=0.0.0.0:9000"
                "RUSTFS_CONSOLE_ENABLE=true"
                "RUSTFS_CONSOLE_ADDRESS=0.0.0.0:9001"
                "RUSTFS_ACCESS_KEY=runflow"
                "RUSTFS_SECRET_KEY=runflow-secret"
                "RUSTFS_BUCKET=runflow"
                "AWS_ACCESS_KEY_ID=runflow"
                "AWS_SECRET_ACCESS_KEY=runflow-secret"
                "AWS_REGION=us-east-1"
                "AWS_ENDPOINT_URL=http://127.0.0.1:9000"
                "BACKEND_PORT=8000"
                "RUNNER_ID=runner-1"
              ];

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
