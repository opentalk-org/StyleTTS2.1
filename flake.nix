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
          ps.websockets
        ];

      runnerDependencies = ps:
        runflowDependencies ps ++ [
          ps."nats-py"
        ];

      mkFrontendStatic = pkgs: pkgs.buildNpmPackage {
        pname = "runflow-studio-frontend";
        version = "0.1.0";
        src = ./src/frontend;
        npmDepsHash = "sha256-dIwSK4LCcKc8IX2hgSyUIctHuygcoKK9tbpm2b9y6bI=";
        installPhase = ''
          runHook preInstall
          mkdir -p $out
          cp -r dist/* $out/
          runHook postInstall
        '';
      };

      mkDevShell = system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          backendEnv = python.withPackages backendDependencies;
          runnerEnv = python.withPackages runnerDependencies;
          runflowEnv = python.withPackages runflowDependencies;
          runflowDev = pkgs.writeShellApplication {
            name = "runflow-dev";
            runtimeInputs = [
              backendEnv
              runnerEnv
              pkgs.bash
              pkgs.coreutils
              pkgs.nats-server
              pkgs.nodejs_22
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
        in
        pkgs.mkShell {
          packages = [
            backendEnv
            runnerEnv
            runflowEnv
            runflowDev
            frontendDev
            pkgs.nats-server
            pkgs.nodejs_22
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
      frontendStatic = mkFrontendStatic pkgs;

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
        in
        {
          frontend-static = mkFrontendStatic pkgsForSystem;
          default = self.packages.${system}.frontend-static;
        }
      ) // {
        ${imageSystem} = rec {
          frontend-static = frontendStatic;

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
            frontendStatic
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
      };
      defaultPackage.${imageSystem} = self.packages.${imageSystem}.image;
    };
}
