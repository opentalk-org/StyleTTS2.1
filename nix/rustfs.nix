{ pkgs }:

let
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
}
