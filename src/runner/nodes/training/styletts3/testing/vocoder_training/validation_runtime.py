from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def validation_cudnn_benchmark_disabled() -> Iterator[None]:
    """Avoid per-shape autotuning for variable-length validation recordings."""
    enabled = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = enabled
