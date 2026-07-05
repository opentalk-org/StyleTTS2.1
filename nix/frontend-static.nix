{ pkgs }:

pkgs.buildNpmPackage {
  pname = "runflow-studio-frontend";
  version = "0.1.0";
  src = ../src/frontend;
  npmDepsHash = "sha256-dIwSK4LCcKc8IX2hgSyUIctHuygcoKK9tbpm2b9y6bI=";
  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r dist/* $out/
    runHook postInstall
  '';
}
