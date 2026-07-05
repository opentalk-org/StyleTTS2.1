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

      rustfsVersion = "1.0.0-beta.8";
      rustfsAssets = {
        aarch64-darwin = {
          asset = "rustfs-macos-aarch64-v${rustfsVersion}.zip";
          hash = "1vc7g8db66d7q9krpihm31qynka26256jp1pljrq2w991lbf90bd";
        };
        aarch64-linux = {
          asset = "rustfs-linux-aarch64-musl-v${rustfsVersion}.zip";
          hash = "0ashi7m1rgfxl6jg7wrsvlidw9adzzf4cg0yqa2lyqzcf18g80vh";
        };
        x86_64-darwin = {
          asset = "rustfs-macos-x86_64-v${rustfsVersion}.zip";
          hash = "0fi8y1bz1q7df6gkl8nnahky6jh84k30yz365qw0rpp5kalxxrsf";
        };
        x86_64-linux = {
          asset = "rustfs-linux-x86_64-musl-v${rustfsVersion}.zip";
          hash = "1bxffbxar04frcvibh01s9iigzpkhwgk4p18pi1axi1yya1gqxc1";
        };
      };

      mkRustfs = pkgs:
        let
          asset = rustfsAssets.${pkgs.system};
        in
        pkgs.stdenvNoCC.mkDerivation {
          pname = "rustfs";
          version = rustfsVersion;
          src = pkgs.fetchurl {
            url = "https://github.com/rustfs/rustfs/releases/download/${rustfsVersion}/${asset.asset}";
            sha256 = asset.hash;
          };
          dontUnpack = true;
          nativeBuildInputs = [ pkgs.unzip ];
          installPhase = ''
            runHook preInstall
            if unzip -t "$src" >/dev/null 2>&1; then
              unzip -q "$src" rustfs
              install -Dm755 rustfs "$out/bin/rustfs"
            else
              install -Dm755 "$src" "$out/bin/rustfs"
            fi
            runHook postInstall
          '';
        };

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
          rustfs = mkRustfs pkgs;
          runflowDev = pkgs.writeShellApplication {
            name = "runflow-dev";
            runtimeInputs = [
              pkgs.awscli2
              backendEnv
              runnerEnv
              pkgs.bash
              pkgs.coreutils
              pkgs.nats-server
              pkgs.nodejs_22
              pkgs.pgbouncer
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
        in
        pkgs.mkShell {
          packages = [
            pkgs.awscli2
            backendEnv
            runnerEnv
            runflowEnv
            runflowDev
            frontendDev
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
          exec python -m runner.worker \
            --runner-id "''${RUNNER_ID:-runner-1}" \
            --nats-url "''${NATS_URL:-nats://127.0.0.1:4222}"
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
