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
    rustfs.url = "github:rustfs/rustfs/1.0.0-beta.8";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    {
      nixpkgs,
      rustfs,
      flake-parts,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
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

          packages = {
            frontend-static = pkgs.callPackage ./nix/frontend-static.nix;
            default = self'.packages.frontend-static;
          };

          devShells.default = pkgs.mkShell {
            packages = [
              rustfs.packages.${system}.default
              python

              pkgs.awscli2
              pkgs.nodejs_22
              pkgs.openssh
              pkgs.pgbouncer
              pkgs.postgresql_16
              pkgs.rclone
              pkgs.uv
              pkgs.zellij
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
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
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
              export PYTHONPATH="$PWD/src"
              if [ -e .venv/bin/activate ]; then
                . .venv/bin/activate
              else
                echo "no .venv yet - run: uv sync --frozen"
              fi
            '';
          };
        };
    };
}
