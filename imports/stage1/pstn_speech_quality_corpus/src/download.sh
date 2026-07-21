#!/usr/bin/env bash
set -euo pipefail

dataset_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
archive="$dataset_dir/tmp/train.zip"

mkdir -p "$dataset_dir/tmp" "$dataset_dir/wavs"
curl --fail --location --continue-at - \
    --output "$archive" \
    "https://challenge.blob.core.windows.net/pstn/train.zip"
