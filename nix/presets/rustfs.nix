{
  name,
  config,
  lib,
  pkgs,
  dnvrState,
  inputs,
  system,
  ...
}:
let
  inherit (lib) mkOption types;
  presetLib = import "${inputs.dnvr}/presets/lib.nix" { inherit lib; };
in
{
  options = {
    bucket = mkOption {
      type = types.str;
      description = ''
        Bucket created (if missing) once the server is up. Published as
        `bucket` only after it exists, so `dnvr://<name>/bucket` doubles as a
        readiness signal for consumers that write objects.
      '';
    };
    extraBuckets = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Additional buckets created alongside `bucket`.";
    };
    address = mkOption {
      type = types.str;
      default = "127.0.0.1";
    };
    port = mkOption {
      type = types.port;
      default = 9000;
    };
    consoleEnable = mkOption {
      type = types.bool;
      default = true;
    };
    consolePort = mkOption {
      type = types.port;
      default = 9001;
    };
    accessKey = mkOption {
      type = types.str;
      default = "runflow";
    };
    secretKey = mkOption {
      type = types.str;
      default = "runflow-secret";
    };
    region = mkOption {
      type = types.str;
      default = "us-east-1";
    };
    dataDir = mkOption {
      type = types.str;
      default = ".dnvr/${name}";
    };
    logDir = mkOption {
      type = types.str;
      default = ".dnvr/logs";
    };
    package = mkOption {
      type = types.package;
      default = inputs.rustfs.packages.${system}.default;
    };
    awscli = mkOption {
      type = types.package;
      default = pkgs.awscli2;
    };

    # Computed, read-only. Static strings usable anywhere in config — no
    # readiness implied. Paths are `$DNVR_ROOT`-relative shell strings.
    endpoint = mkOption {
      type = types.str;
      readOnly = true;
      description = "S3 API endpoint URL, e.g. for AWS_ENDPOINT_URL.";
    };
    dataPath = mkOption {
      type = types.str;
      readOnly = true;
      description = "Absolute data directory (`$DNVR_ROOT/<dataDir>`).";
    };
  };

  config =
    let
      upper = presetLib.envPrefix name;
      host = presetLib.connectableHost config.address;
      endpoint = "http://${host}:${toString config.port}";
      allBuckets = [ config.bucket ] ++ config.extraBuckets;
      aws = "aws --endpoint-url ${endpoint}";
    in
    {
      endpoint = endpoint;
      dataPath = "$DNVR_ROOT/${config.dataDir}";

      packages = [
        config.package
        config.awscli
      ];

      env = {
        "S3_${upper}_ENDPOINT" = endpoint;
        "S3_${upper}_BUCKET" = config.bucket;
        "S3_${upper}_LOG" = "$DNVR_ROOT/${config.logDir}/${name}.log";
      };

      command = pkgs.writeShellApplication {
        name = "${name}-rustfs";
        runtimeInputs = [
          config.package
          config.awscli
          pkgs.coreutils
          dnvrState
        ];
        text = ''
          : "''${DNVR_ROOT:?DNVR_ROOT must be set}"
          mkdir -p "$DNVR_ROOT/${config.dataDir}" "$DNVR_ROOT/${config.logDir}"

          # Discovery keys, published before rustfs is listening; the readiness
          # keys (bucket/bucketUrl) follow once the API answers and the buckets
          # exist.
          dnvr-state set host "${host}"
          dnvr-state set port "${toString config.port}"
          dnvr-state set endpoint "${endpoint}"
          dnvr-state set accessKey ${lib.escapeShellArg config.accessKey}
          dnvr-state set secretKey ${lib.escapeShellArg config.secretKey}
          dnvr-state set region ${lib.escapeShellArg config.region}
          ${lib.optionalString config.consoleEnable ''
            dnvr-state set consoleUrl "http://${host}:${toString config.consolePort}"
          ''}

          echo "[${name}] starting rustfs at ${endpoint} ..."
          rustfs server \
            --address "${host}:${toString config.port}" \
            --access-key ${lib.escapeShellArg config.accessKey} \
            --secret-key ${lib.escapeShellArg config.secretKey} \
            --console-address "${host}:${toString config.consolePort}" \
            ${lib.optionalString config.consoleEnable "--console-enable"} \
            "$DNVR_ROOT/${config.dataDir}" &
          RUSTFS_PID=$!
          # `wait` (not a foreground run) so a trapped signal is handled
          # immediately instead of being deferred until rustfs returns.
          trap '
            kill -TERM $RUSTFS_PID 2>/dev/null || true
            wait $RUSTFS_PID 2>/dev/null || true
          ' EXIT INT TERM

          # The aws cli reads credentials from the environment; scope them to
          # this wrapper so the devshell's own AWS_* stay untouched.
          export AWS_ACCESS_KEY_ID=${lib.escapeShellArg config.accessKey}
          export AWS_SECRET_ACCESS_KEY=${lib.escapeShellArg config.secretKey}
          export AWS_DEFAULT_REGION=${lib.escapeShellArg config.region}
          export AWS_EC2_METADATA_DISABLED=true

          ${presetLib.untilReady {
            pid = "$RUSTFS_PID";
            check = "${aws} s3api list-buckets >/dev/null 2>&1";
            onDead = "[${name}] rustfs exited before becoming ready";
            interval = "0.2";
          }}

          # Buckets are created before the readiness key is published, so a
          # consumer waiting on `dnvr://${name}/bucket` cannot observe a
          # running server with a missing bucket.
          ${lib.concatMapStrings (bucket: ''
            if ! ${aws} s3api head-bucket --bucket ${lib.escapeShellArg bucket} >/dev/null 2>&1; then
              echo "[${name}] creating bucket ${bucket} ..."
              ${aws} s3api create-bucket --bucket ${lib.escapeShellArg bucket} >/dev/null
            fi
          '') allBuckets}

          dnvr-state set bucket ${lib.escapeShellArg config.bucket}
          dnvr-state set bucketUrl "s3://${config.bucket}"
          echo "[${name}] ready — ${endpoint}, bucket ${config.bucket}"

          wait $RUSTFS_PID
        '';
      };
    };
}
