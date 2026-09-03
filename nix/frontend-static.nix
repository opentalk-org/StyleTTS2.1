{ pkgs }:

pkgs.buildNpmPackage {
  pname = "runflow-studio-frontend";
  version = "0.1.0";
  src = ../src/frontend;
  npmDepsHash = "sha256-zuOGBd+wOiCWyZQIAigbubDHE100icEMKSxdKMoKG/A=";
  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r dist/client/* $out/
    runHook postInstall
  '';
}
