from collections.abc import Iterable, Mapping

import torch
from accelerate import Accelerator
from torch import nn


def gradient_norm_metrics(
    accelerator: Accelerator,
    modules: Mapping[str, nn.Module],
    module_names: Iterable[str],
    group_name: str | None = None,
) -> dict[str, torch.Tensor | float]:
    gradient_scale = (
        float(accelerator.scaler.get_scale())
        if accelerator.scaler is not None
        else 1.0
    )
    norms = {
        name: _module_gradient_norm(modules[name], gradient_scale)
        for name in module_names
    }
    metrics = {
        f"gradient_norm/{name}": norm
        for name, norm in norms.items()
    }
    if group_name is not None:
        metrics[f"gradient_norm/{group_name}"] = _combined_norm(norms.values())
    return metrics


def _module_gradient_norm(
    module: nn.Module,
    gradient_scale: float,
) -> torch.Tensor | float:
    parameter_norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float())
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not parameter_norms:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(parameter_norms)) / gradient_scale


def _combined_norm(
    norms: Iterable[torch.Tensor | float],
) -> torch.Tensor | float:
    tensors = [norm for norm in norms if isinstance(norm, torch.Tensor)]
    scalar_squares = sum(norm * norm for norm in norms if isinstance(norm, float))
    if not tensors:
        return scalar_squares**0.5
    tensor_squares = torch.stack([norm.square() for norm in tensors]).sum()
    return (tensor_squares + scalar_squares).sqrt()
