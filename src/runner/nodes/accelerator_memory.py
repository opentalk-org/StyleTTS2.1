from __future__ import annotations

import gc
from typing import Any

import torch


def maybe_cuda_half(model: Any) -> Any:
    if torch.cuda.is_available():
        return model.cuda().half()
    return model


def release_accelerator_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
