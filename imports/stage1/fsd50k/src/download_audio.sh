#!/usr/bin/env bash
set -euo pipefail

dataset_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
record_url="https://zenodo.org/api/records/4060432/files"
mkdir -p "$dataset_dir/tmp" "$dataset_dir/wavs"

files=(
    FSD50K.eval_audio.z01
    FSD50K.eval_audio.zip
    FSD50K.dev_audio.z01
    FSD50K.dev_audio.z02
    FSD50K.dev_audio.z03
    FSD50K.dev_audio.z04
    FSD50K.dev_audio.z05
    FSD50K.dev_audio.zip
)
for filename in "${files[@]}"; do
    curl --fail --location --continue-at - \
        --output "$dataset_dir/tmp/$filename" \
        "$record_url/$filename/content"
done
