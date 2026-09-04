import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import Tensor, nn

HISTOGRAM_BINS = 64


class ParameterMonitor:
    def __init__(
        self,
        accelerator: Accelerator,
        modules: Mapping[str, nn.Module],
        every_steps: int,
        initial_step: int,
    ) -> None:
        self._accelerator = accelerator
        self._modules = modules
        self._every_steps = every_steps
        self._steps = {name: initial_step for name in modules}
        self._metrics: dict[str, float | list[float]] = {}

    def __call__(self, module_name: str) -> None:
        step = self._steps[module_name] + 1
        self._steps[module_name] = step
        if step % self._every_steps == 0:
            self._metrics.update(
                parameter_histogram_metrics(
                    self._accelerator,
                    self._modules,
                    (module_name,),
                )
            )

    def drain(self) -> dict[str, float | list[float]]:
        metrics = self._metrics
        self._metrics = {}
        return metrics


def write_model_graph(modules: Mapping[str, nn.Module], path: Path) -> None:
    records: list[dict[str, str | int | None | list[str]]] = []
    visited: set[int] = set()
    for root_name, root in modules.items():
        for relative_name, module in root.named_modules():
            if id(module) in visited:
                continue
            visited.add(id(module))
            component_id = _component_path(root_name, relative_name)
            parent_id = component_id.rsplit(".", 1)[0] if "." in component_id else None
            parameters = [name for name, _ in module.named_parameters(recurse=False)]
            records.append({
                "id": component_id,
                "parent_id": parent_id,
                "name": component_id.rsplit(".", 1)[-1],
                "module_type": type(module).__name__,
                "parameter_names": parameters,
                "parameter_count": sum(
                    parameter.numel()
                    for parameter in module.parameters(recurse=False)
                ),
            })
    path.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")


def parameter_histogram_metrics(
    accelerator: Accelerator,
    modules: Mapping[str, nn.Module],
    module_names: Iterable[str],
) -> dict[str, float | list[float]]:
    gradient_scale = (
        float(accelerator.scaler.get_scale())
        if accelerator.scaler is not None
        else 1.0
    )
    metrics: dict[str, float | list[float]] = {}
    visited: set[int] = set()
    for module_name in module_names:
        for parameter_name, parameter in modules[module_name].named_parameters():
            if id(parameter) in visited or not parameter.requires_grad:
                continue
            visited.add(id(parameter))
            path = f"{module_name}.{parameter_name}"
            metrics[f"param/{path}"] = _histogram(parameter.detach())
            metrics[f"param_nonfinite/{path}"] = _nonfinite_fraction(parameter.detach())
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float() / gradient_scale
                metrics[f"grad/{path}"] = _histogram(gradient)
                metrics[f"grad_nonfinite/{path}"] = _nonfinite_fraction(gradient)
    return metrics


def _histogram(tensor: Tensor) -> list[float]:
    values = tensor.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return [0.0, 0.0, *([0.0] * HISTOGRAM_BINS)]
    lower = float(finite.min().item())
    upper = float(finite.max().item())
    if lower == upper:
        counts = [float(finite.numel()), *([0.0] * (HISTOGRAM_BINS - 1))]
        return [lower, upper, *counts]
    counts = torch.histc(finite, bins=HISTOGRAM_BINS, min=lower, max=upper)
    return [lower, upper, *counts.cpu().tolist()]


def _nonfinite_fraction(tensor: Tensor) -> float:
    return float((~torch.isfinite(tensor)).float().mean().item())


def _component_path(root_name: str, relative_name: str) -> str:
    return root_name if relative_name == "" else f"{root_name}.{relative_name}"
