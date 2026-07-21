#!/usr/bin/env bash
set -euo pipefail

dataset_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_url="https://huggingface.co/datasets/amphion/Emilia-NV/resolve/main"

mkdir -p "$dataset_dir/tmp" "$dataset_dir/wavs"
source "$dataset_dir/../../../.env"

for shard_index in $(seq -w 0 17); do
    shard_name="webdataset-0000${shard_index}.tar"
    curl --fail --location --continue-at - \
        --header "Authorization: Bearer ${HF_TOKEN}" \
        --output "$dataset_dir/tmp/$shard_name" \
        "$repo_url/$shard_name"
done
