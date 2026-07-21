#!/usr/bin/env bash
set -euo pipefail

dataset_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$dataset_dir/src/download.sh"
python "$dataset_dir/src/prepare.py"
