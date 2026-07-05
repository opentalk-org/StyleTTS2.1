{
  description = "Runflow Studio single-image runtime with backend, runner, NATS JetStream, and PostgreSQL";

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

      runflowDependencies = ps: [
        ps.pydantic
      ];

      backendDependencies = ps:
        runflowDependencies ps ++ [
          ps.fastapi
          ps."nats-py"
          ps.uvicorn
        ];

      runnerDependencies = ps:
        runflowDependencies ps ++ [
          ps."nats-py"
        ];

      mkDevShell = system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          backendEnv = python.withPackages backendDependencies;
          runnerEnv = python.withPackages runnerDependencies;
          runflowEnv = python.withPackages runflowDependencies;
        in
        pkgs.mkShell {
          packages = [
            backendEnv
            runnerEnv
            runflowEnv
            pkgs.nats-server
            pkgs.postgresql_16
          ];

          shellHook = ''
            export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
          '';
        };

      pkgs = import nixpkgs { system = imageSystem; };
      python = pkgs.python312;
      runflowEnv = python.withPackages runflowDependencies;
      backendEnv = python.withPackages backendDependencies;
      runnerEnv = python.withPackages runnerDependencies;

      runflowBackend = pkgs.writeShellApplication {
        name = "runflow-backend";
        runtimeInputs = [ backendEnv ];
        text = ''
          cd ${./.}
          export PYTHONPATH="${./src}:''${PYTHONPATH:-}"
          exec uvicorn backend.api:app --host 0.0.0.0 --port "''${BACKEND_PORT:-8000}"
        '';
      };

      runflowRunner = pkgs.writeShellApplication {
        name = "runflow-runner";
        runtimeInputs = [ runnerEnv ];
        text = ''
          cd ${./.}
          export PYTHONPATH="${./src}:''${PYTHONPATH:-}"
          exec python -m runner.worker \
            --runner-id "''${RUNNER_ID:-runner-1}" \
            --nats-url "''${NATS_URL:-nats://127.0.0.1:4222}"
        '';
      };

      entrypoint = pkgs.writeShellApplication {
        name = "runflow-entrypoint";
        runtimeInputs = [
          pkgs.bash
          pkgs.coreutils
          pkgs.gnugrep
          pkgs.gnused
          pkgs.nats-server
          pkgs.postgresql_16
          runflowBackend
          runflowRunner
        ];
        text = ''
          set -euo pipefail

          export PGDATA="''${PGDATA:-/data/postgres}"
          export PGHOST="''${PGHOST:-/tmp/postgres}"
          export PGPORT="''${PGPORT:-5432}"
          export POSTGRES_DB="''${POSTGRES_DB:-runflow}"
          export POSTGRES_USER="''${POSTGRES_USER:-runflow}"
          export POSTGRES_PASSWORD="''${POSTGRES_PASSWORD:-runflow}"
          export NATS_DATA="''${NATS_DATA:-/data/nats}"
          export NATS_URL="''${NATS_URL:-nats://127.0.0.1:4222}"
          export BACKEND_PORT="''${BACKEND_PORT:-8000}"
          export DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@127.0.0.1:$PGPORT/$POSTGRES_DB"

          mkdir -p "$PGDATA" "$PGHOST" "$NATS_DATA"

          if [ ! -s "$PGDATA/PG_VERSION" ]; then
            echo "Initializing PostgreSQL database at $PGDATA"
            initdb -D "$PGDATA" --auth=trust

            cat >> "$PGDATA/postgresql.conf" <<EOF
          listen_addresses = '*'
          port = $PGPORT
          unix_socket_directories = '$PGHOST'
          EOF

            cat >> "$PGDATA/pg_hba.conf" <<EOF
          host all all 0.0.0.0/0 md5
          host all all ::/0 md5
          EOF
          fi

          echo "Starting PostgreSQL"
          pg_ctl -D "$PGDATA" \
            -o "-k $PGHOST -p $PGPORT" \
            -l /tmp/postgres.log \
            start

          until pg_isready -h "$PGHOST" -p "$PGPORT"; do
            sleep 1
          done

          echo "Ensuring PostgreSQL database/user exist"
          psql -h "$PGHOST" -p "$PGPORT" -d postgres <<SQL
          DO \$\$
          BEGIN
            IF NOT EXISTS (
              SELECT FROM pg_catalog.pg_roles WHERE rolname = '$POSTGRES_USER'
            ) THEN
              CREATE ROLE "$POSTGRES_USER" LOGIN PASSWORD '$POSTGRES_PASSWORD';
            END IF;
          END
          \$\$;
          SQL

          if ! psql -h "$PGHOST" -p "$PGPORT" -d postgres -tAc \
            "SELECT 1 FROM pg_database WHERE datname = '$POSTGRES_DB'" \
            | grep -q 1; then
            createdb -h "$PGHOST" -p "$PGPORT" -O "$POSTGRES_USER" "$POSTGRES_DB"
          fi

          echo "Starting NATS JetStream"
          nats-server -js -sd "$NATS_DATA" -p 4222 > /tmp/nats.log 2>&1 &
          pid_nats=$!

          until grep -q "Server is ready" /tmp/nats.log 2>/dev/null; do
            if ! kill -0 "$pid_nats" 2>/dev/null; then
              echo "NATS exited before becoming ready"
              exit 1
            fi
            sleep 1
          done

          echo "Starting Runflow backend"
          runflow-backend &
          pid_backend=$!

          echo "Starting Runflow runner"
          runflow-runner &
          pid_runner=$!

          shutdown() {
            echo "Stopping services"
            kill "$pid_backend" "$pid_runner" "$pid_nats" 2>/dev/null || true
            pg_ctl -D "$PGDATA" stop -m fast || true
          }

          trap shutdown EXIT INT TERM
          wait -n "$pid_backend" "$pid_runner" "$pid_nats"
        '';
      };
    in
    {
      devShells = forAllDevSystems (system: {
        default = mkDevShell system;
      });

      packages.${imageSystem} = rec {
        image = pkgs.dockerTools.buildLayeredImage {
          name = "runflow-studio-single-image";
          tag = "latest";

          contents = [
            pkgs.bash
            pkgs.cacert
            pkgs.coreutils
            pkgs.nats-server
            pkgs.postgresql_16
            runflowEnv
            runflowBackend
            runflowRunner
            entrypoint
          ];

          config = {
            Cmd = [ "${entrypoint}/bin/runflow-entrypoint" ];

            ExposedPorts = {
              "8000/tcp" = {};
              "4222/tcp" = {};
              "5432/tcp" = {};
            };

            Env = [
              "PGDATA=/data/postgres"
              "PGHOST=/tmp/postgres"
              "PGPORT=5432"
              "POSTGRES_DB=runflow"
              "POSTGRES_USER=runflow"
              "POSTGRES_PASSWORD=runflow"
              "NATS_DATA=/data/nats"
              "NATS_URL=nats://127.0.0.1:4222"
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
      defaultPackage.${imageSystem} = self.packages.${imageSystem}.image;
    };
}
