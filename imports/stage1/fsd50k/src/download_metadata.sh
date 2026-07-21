#!/usr/bin/env bash
set -euo pipefail

dataset_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record_url="https://zenodo.org/api/records/4060432/files"
mkdir -p "$dataset_dir/tmp" "$dataset_dir/wavs"

for filename in FSD50K.ground_truth.zip FSD50K.metadata.zip FSD50K.doc.zip; do
    curl --fail --location --continue-at - \
        --output "$dataset_dir/tmp/$filename" \
        "$record_url/$filename/content"
done
