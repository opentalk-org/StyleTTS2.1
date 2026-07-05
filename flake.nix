{
  description = "Runflow Studio single-image runtime with backend, runner, NATS JetStream, PgBouncer, and PostgreSQL";

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

      mkRustfs = pkgs: import ./nix/rustfs-package.nix { inherit pkgs; };

      runflowDependencies = ps: [
        ps.pydantic
      ];

      backendDependencies = ps:
        runflowDependencies ps ++ [
          ps.boto3
          ps.fastapi
          ps."nats-py"
          ps.psycopg
          ps."python-multipart"
          ps.sqlalchemy
          ps.uvicorn
          ps.websockets
        ];

      runnerDependencies = ps:
        runflowDependencies ps ++ [
          ps."nats-py"
        ];

      mkFrontendStatic = pkgs: import ./nix/frontend-static.nix { inherit pkgs; };

      mkDevShell = system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          backendEnv = python.withPackages backendDependencies;
          runnerEnv = python.withPackages runnerDependencies;
          runflowEnv = python.withPackages runflowDependencies;
          rustfs = mkRustfs pkgs;
          runflowDev = pkgs.writeShellApplication {
            name = "runflow-dev";
            runtimeInputs = [
              pkgs.awscli2
              backendEnv
              runnerEnv
              pkgs.bash
              pkgs.coreutils
              pkgs.gnugrep
              pkgs.nats-server
              pkgs.nodejs_22
              pkgs.pgbouncer
              pkgs.postgresql_16
              rustfs
            ];
            text = builtins.readFile ./nix/runflow-dev.sh;
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
            runtimeInputs = [ runnerEnv pkgs.bash pkgs.coreutils pkgs.gnugrep ];
            text = builtins.readFile ./nix/runner-launch.sh;
          };
        in
        pkgs.mkShell {
          packages = [
            pkgs.awscli2
            backendEnv
            runnerEnv
            runflowEnv
            runflowDev
            frontendDev
            runnerLaunch
            pkgs.nats-server
            pkgs.nodejs_22
            pkgs.pgbouncer
            pkgs.postgresql_16
            rustfs
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
      frontendStatic = mkFrontendStatic pkgs;
      rustfs = mkRustfs pkgs;

      runflowBackend = pkgs.writeShellApplication {
        name = "runflow-backend";
        runtimeInputs = [ backendEnv ];
        text = ''
          cd ${./.}
          export PYTHONPATH="${./src}:''${PYTHONPATH:-}"
          export RUNFLOW_UI_STATIC_DIR="${frontendStatic}"
          exec uvicorn backend.api:app --host 0.0.0.0 --port "''${BACKEND_PORT:-8000}"
        '';
      };

      runflowRunner = pkgs.writeShellApplication {
        name = "runflow-runner";
        runtimeInputs = [ runnerEnv ];
        text = ''
          cd ${./.}
          export PYTHONPATH="${./src}:''${PYTHONPATH:-}"
          exec python -m runner.cli \
            --runner-id "''${RUNNER_ID:-runner-1}" \
            --nats-url "''${NATS_URL:-nats://127.0.0.1:4222}"
        '';
      };
      runnerLaunch = pkgs.writeShellApplication {
        name = "runflow-runner-launch";
        runtimeInputs = [ runnerEnv pkgs.bash pkgs.coreutils pkgs.gnugrep runflowRunner ];
        text = builtins.readFile ./nix/runner-launch.sh;
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
        ];
        text = builtins.readFile ./nix/entrypoint.sh;
      };
    in
    {
      devShells = forAllDevSystems (system: {
        default = mkDevShell system;
      });

      packages = forAllDevSystems (system:
        let
          pkgsForSystem = import nixpkgs { inherit system; };
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
            runflowEnv
            frontendStatic
            runflowBackend
            runflowRunner
            entrypoint
          ];

          config = {
            Cmd = [ "${entrypoint}/bin/runflow-entrypoint" ];

            ExposedPorts = {
              "8000/tcp" = {};
              "9000/tcp" = {};
              "9001/tcp" = {};
              "4222/tcp" = {};
              "6432/tcp" = {};
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
