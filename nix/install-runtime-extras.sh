#!/usr/bin/env bash
# Optional runtime extras that are NOT in pyproject/uv.lock because their pinned
# metadata conflicts with the locked resolution (DeepFilterNet pins numpy<2 while
# the rest of the stack needs numpy>=2). They import and run fine against the
# locked environment, so we install them with --no-deps on top of `uv sync`.
#
# Run this once after `uv sync` (or after any `uv sync --frozen` that pruned the
# environment):  bash nix/install-runtime-extras.sh
set -euo pipefail

PYTHON="${1:-.venv/bin/python}"

# DeepFilterNet denoise node (df / libdf) + its light runtime deps.
uv pip install --python "$PYTHON" --no-deps \
  deepfilternet==0.5.6 deepfilterlib==0.5.6
uv pip install --python "$PYTHON" loguru appdirs

# StyleTTS2 monotonic alignment (Cython core used by the finetune loop).
uv pip install --python "$PYTHON" monotonic-align

echo "runtime extras installed"
