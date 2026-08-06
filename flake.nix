{
  description = "Runflow Studio runtime with PostgreSQL-coordinated backend and runners";

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
    flake-parts.url = "github:hercules-ci/flake-parts";
    dnvr.url = "github:dialohq/dnvr";
  };

  outputs =
    {
      nixpkgs,
      flake-parts,
      dnvr,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ dnvr.flakeModule ];
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      perSystem =
        {
          self',
          pkgs,
          system,
          lib,
          presets,
          ...
        }:
        let
          runtimeLibs = [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];
          runtimeExecutableDeps = [
            espeakNg
            pkgs.ffmpeg-headless
            pkgs.opusTools
            pkgs.gcc
            pkgs.cargo
            pkgs.openssl
            pkgs.patchelf
            pkgs.rustc
          ];
          cuda = pkgs.cudaPackages.cuda_nvcc;
          python = pkgs.python312Full;
          linuxDeps = [
            pkgs.glibc
            cuda
          ];
          nvidiaDriverPath = pkgs.lib.concatStringsSep ":" [
            "/usr/local/nvidia/lib"
            "/usr/local/nvidia/lib64"
          ];
        in
        {
          formatter = pkgs.nixfmt;

          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [
              (final: prev: {
                espeak-ng = prev.espeak-ng.overrideAttrs (_: {
                  version = "1.52.0-unstable-2025-09-08";
                  patches = [ ];
                  src = final.fetchFromGitHub {
                    owner = "espeak-ng";
                    repo = "espeak-ng";
                    rev = "0d451f8c1c6ae837418b823bd9c4cbc574ea9ff5";
                    hash = "sha256-wpPi+YjSLhsEWfE3KEbL4A7o48qtz9fLRZ/u4xGOM2g=";
                  };
                });
              })
            ];
          };

          dnvr.specialArgs = { inherit inputs system; };
          dnvr.presets.rustfs = ./nix/presets/rustfs.nix;

          packages = {
            frontend-static = pkgs.callPackage ./nix/frontend-static.nix;
            default = self'.packages.frontend-static;
          };

          dnvr.shells.default = { config, ... }: {
            packages = [
              (import ./nix/rustfs.nix { inherit pkgs; })
              python

              pkgs.awscli2
              pkgs.nodejs_22
              pkgs.openssh
              pkgs.pgbouncer
              pkgs.postgresql_16
              pkgs.rclone
              pkgs.uv
              pkgs.pyright
            ]
            ++ runtimeLibs
            ++ runtimeExecutableDeps
            ++ (if pkgs.stdenv.isLinux then linuxDeps else [ ]);

            env = {
              CC = lib.getExe pkgs.gcc;
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              PHONEMIZER_ESPEAK_LIBRARY = "${pkgs.espeak-ng}/lib/libespeak-ng.so";
              PYTHONUNBUFFERED = "1";

              CPATH = "${pkgs.portaudio}/include";

              UV_PYTHON = lib.getExe python;
              UV_PYTHON_PREFERENCE = "only-system";
              UV_PYTHON_DOWNLOADS = "never";

              PYTHONPATH = "$DNVR_ROOT/src";

              AWS_ACCESS_KEY_ID = "runflow";
              AWS_SECRET_ACCESS_KEY = "runflow-secret";
              AWS_REGION = "us-east-1";
              AWS_DEFAULT_REGION = "us-east-1";
              AWS_ENDPOINT_URL = "http://127.0.0.1:9000";

              RUNFLOW_S3_BUCKET = "runflow";
              RUNFLOW_S3_ENDPOINT_URL = "http://127.0.0.1:9000";
              RUNFLOW_S3_REGION = "us-east-1";
              RUNFLOW_S3_ACCESS_KEY_ID = "runflow";
              RUNFLOW_S3_SECRET_ACCESS_KEY = "runflow-secret";

              RUNFLOW_PGBOUNCER_DATABASE_URL = "postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow";
              RUNFLOW_NOTIFY_DATABASE_URL = "postgresql+psycopg://runflow:runflow@127.0.0.1:5432/runflow";

              MLFLOW_TRACKING_URI = "http://127.0.0.1:7860";
              MLFLOW_S3_ENDPOINT_URL = "http://127.0.0.1:9000";
              VITE_BACKEND_URL = "http://127.0.0.1:8001";
              RUNNER_ID = "runner-1";

              HF_HOME = "$DNVR_ROOT/.cache/huggingface";
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True";

              LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.ffmpeg-headless.lib}/lib:${pkgs.portaudio}/lib:${nvidiaDriverPath}";
              LIBRARY_PATH = "${pkgs.portaudio}/lib:${nvidiaDriverPath}";

              TRITON_PTXAS_PATH = "${cuda}/bin/ptxas";
              TRITON_PTXAS_BLACKWELL_PATH = "${cuda}/bin/ptxas";
              TRITON_LIBCUDA_PATH = "/usr/local/nvidia/lib";
            };

            shellHook = ''
              unset NIX_CFLAGS_COMPILE CFLAGS CXXFLAGS
              if [ -f .env ]; then
                set -a
                . ./.env
                set +a
              fi
              if [ -e .venv/bin/activate ]; then
                . .venv/bin/activate
              else
                echo "no .venv yet - run: uv sync --frozen"
              fi
            '';

            processes.pg = {
              imports = [ presets.postgres ];
              package = pkgs.postgresql_16;
              database = "runflow";
              extraDatabases = [ "mlflow" ];
              # SUPERUSER because the preset creates databases owned by the
              # `postgres` superuser, and both alembic migrations and MLflow's
              # own schema bootstrap need to write into them.
              initialScript = ''
                DO $$
                  BEGIN
                    IF NOT EXISTS (
                      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'runflow'
                    ) THEN
                      CREATE ROLE "runflow" LOGIN PASSWORD 'runflow' SUPERUSER;
                    END IF;
                  END
                $$
              '';
            };

            processes.s3 = {
              imports = [ presets.rustfs ];
              bucket = "runflow";
            };

            processes.pgbouncer = {
              env = {
                PGB_SOCKET_DIR = "dnvr://pg/socketDir";
                PGB_PORT = "dnvr://pg/port";
                PGB_DATABASE = "dnvr://pg/database";
              };
              command = pkgs.writeShellApplication {
                name = "pgbouncer-run";
                runtimeInputs = [
                  pkgs.pgbouncer
                  pkgs.postgresql_16
                  pkgs.coreutils
                ];
                text = ''
                  dir="$DNVR_STATE/pgbouncer"
                  mkdir -p "$dir"
                  printf '"runflow" "runflow"\n' > "$dir/userlist.txt"

                  cat > "$dir/pgbouncer.ini" <<EOF
                  [databases]
                  runflow = host=''${PGB_SOCKET_DIR} port=''${PGB_PORT} dbname=''${PGB_DATABASE}

                  [pgbouncer]
                  listen_addr = 127.0.0.1
                  listen_port = 6432
                  unix_socket_dir = $dir
                  auth_type = plain
                  auth_file = $dir/userlist.txt
                  pool_mode = transaction
                  max_client_conn = 200
                  default_pool_size = 20
                  reserve_pool_size = 5
                  ignore_startup_parameters = extra_float_digits
                  pidfile = $dir/pgbouncer.pid
                  EOF

                  rm -f "$dir/pgbouncer.pid"

                  echo "[pgbouncer] starting on 6432 (transaction pooling) ..."
                  pgbouncer "$dir/pgbouncer.ini" &
                  PGB_PID=$!
                  trap '
                    kill -TERM $PGB_PID 2>/dev/null || true
                    wait $PGB_PID 2>/dev/null || true
                  ' EXIT INT TERM

                  until pg_isready -h 127.0.0.1 -p 6432 -d runflow -U runflow >/dev/null 2>&1; do
                    if ! kill -0 $PGB_PID 2>/dev/null; then
                      echo "[pgbouncer] exited before becoming ready" >&2
                      exit 1
                    fi
                    sleep 0.2
                  done

                  # Refs carry whole values only, so the pooled DSN is composed
                  # here rather than by each consumer.
                  dnvr-state set url "postgresql+psycopg://runflow:runflow@127.0.0.1:6432/runflow"
                  dnvr-state set port 6432
                  dnvr-state set database runflow
                  echo "[pgbouncer] ready"

                  wait $PGB_PID
                '';
              };
            };

            processes.mlflow = {
              env = {
                MLFLOW_WAIT_PG = "dnvr://pg/database";
                MLFLOW_BUCKET = "dnvr://s3/bucket";
              };
              command = pkgs.writeShellApplication {
                name = "mlflow-run";
                runtimeInputs = [ pkgs.coreutils ];
                text = ''
                  echo "[mlflow] http://127.0.0.1:7860"
                  exec mlflow server \
                    --host 127.0.0.1 \
                    --port 7860 \
                    --backend-store-uri "postgresql+psycopg://runflow:runflow@127.0.0.1:5432/mlflow" \
                    --artifacts-destination "s3://''${MLFLOW_BUCKET}/mlflow" \
                    --allowed-hosts "localhost:*,127.0.0.1:*" \
                    --x-frame-options NONE
                '';
              };
            };

            processes.backend = {
              env = {
                RUNFLOW_PGBOUNCER_DATABASE_URL = "dnvr://pgbouncer/url";
                RUNFLOW_S3_BUCKET = "dnvr://s3/bucket";
              };
              command = pkgs.writeShellApplication {
                name = "backend-run";
                runtimeInputs = [ pkgs.coreutils ];
                text = ''
                  echo "[backend] http://127.0.0.1:8001 (legacy UI at /ui-old)"
                  exec uvicorn backend.api:app --host 127.0.0.1 --port 8001
                '';
              };
            };

            processes.frontend = {
              command = pkgs.writeShellApplication {
                name = "frontend-run";
                runtimeInputs = [
                  pkgs.nodejs_22
                  pkgs.coreutils
                ];
                text = ''
                  cd "$DNVR_ROOT/src/frontend"
                  npm install --no-audit --no-fund
                  exec npm run dev -- --host 127.0.0.1 --port 5173
                '';
              };
            };

            processes.runner = {
              env = {
                RUNFLOW_PGBOUNCER_DATABASE_URL = "dnvr://pgbouncer/url";
                RUNFLOW_S3_BUCKET = "dnvr://s3/bucket";
              };
              command = pkgs.writeShellApplication {
                name = "runner-run";
                runtimeInputs = [ pkgs.coreutils ];
                text = ''
                  cd "$DNVR_ROOT"
                  runner_id="''${RUNNER_ID:-runner-1}"
                  echo "[runner] starting $runner_id"
                  exec python -m runner.cli --runner-id "$runner_id"
                '';
              };
            };

            scripts.rclone-s3 = {
              description = "Serve the Hetzner storage box as S3 on :8002";
              runtimeInputs = [
                pkgs.rclone
                pkgs.openssh
              ];
              text = ''
                exec rclone serve s3 \
                  --addr 127.0.0.1:8002 \
                  --auth-key "runflow,runflow-secret" \
                  --sftp-ssh "ssh hetzner-storagebox" \
                  --sftp-disable-hashcheck \
                  --sftp-chunk-size 255Ki \
                  ":sftp:/home/storagebucket"
              '';
            };
          };
        };
    };
}
