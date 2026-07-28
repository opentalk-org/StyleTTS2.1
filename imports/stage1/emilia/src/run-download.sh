#!/usr/bin/env bash
set -euo pipefail

nix develop --command python -m imports.stage1.emilia.src.download
