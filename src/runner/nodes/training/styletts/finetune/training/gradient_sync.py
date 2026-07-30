from collections.abc import Iterable

import torch
from accelerate import Accelerator
from torch import nn


GRADIENT_BUCKET_BYTES = 25 * 1024 * 1024


def synchronize_gradients(
    accelerator: Accelerator,
    modules: dict[str, nn.Module],
    module_names: Iterable[str],
) -> None:
    if accelerator.num_processes == 1:
        return
    buckets: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    bucket_bytes: dict[tuple[torch.device, torch.dtype], int] = {}
    for name in module_names:
        for parameter in modules[name].parameters():
            gradient = parameter.grad
            if gradient is None:
                if not parameter.requires_grad:
                    continue
                gradient = torch.zeros_like(parameter)
                parameter.grad = gradient
            key = (gradient.device, gradient.dtype)
            size = gradient.numel() * gradient.element_size()
            bucket = buckets.setdefault(key, [])
            current_bytes = bucket_bytes.setdefault(key, 0)
            if bucket and current_bytes + size > GRADIENT_BUCKET_BYTES:
                _reduce_bucket(accelerator, bucket)
                bucket.clear()
                current_bytes = 0
            bucket.append(gradient)
            bucket_bytes[key] = current_bytes + size
    for bucket in buckets.values():
        _reduce_bucket(accelerator, bucket)


def _reduce_bucket(
    accelerator: Accelerator,
    gradients: list[torch.Tensor],
) -> None:
    if not gradients:
        return
    flattened = torch.cat([gradient.reshape(-1) for gradient in gradients])
    reduced = accelerator.reduce(flattened, reduction="mean")
    offset = 0
    for gradient in gradients:
        length = gradient.numel()
        gradient.copy_(reduced[offset : offset + length].view_as(gradient))
        offset += length
