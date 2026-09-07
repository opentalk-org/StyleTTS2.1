import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from torch.utils._pytree import tree_leaves


class ExecutionGraphRecorder:
    """Records the leaf-module DAG executed by one real forward pass."""

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        self._records: list[dict[str, Any]] = []
        self._producers: dict[int, str] = {}
        self._gradient_producers: dict[Any, str] = {}
        self._calls: dict[str, int] = {}
        self._operations: dict[str, int] = {}
        self._handles = []
        visited: set[int] = set()
        for root_name, root in modules.items():
            for relative_name, module in root.named_modules():
                if len(tuple(module.children())) > 0 or id(module) in visited:
                    continue
                visited.add(id(module))
                path = root_name if relative_name == "" else f"{root_name}.{relative_name}"
                self._handles.append(
                    module.register_forward_hook(
                        partial(self._record, path),
                        with_kwargs=True,
                    )
                )

    def _record(self, module_path: str, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        invocation = self._calls.get(module_path, 0) + 1
        self._calls[module_path] = invocation
        component_id = module_path if invocation == 1 else f"{module_path}@{invocation}"
        inputs = _tensors((args, kwargs))
        outputs = _tensors(output)
        input_ids = list(dict.fromkeys(
            producer for tensor in inputs
            if (producer := self._producer_for_tensor(tensor)) is not None
        ))
        parameters = [name for name, _ in module.named_parameters(recurse=False)]
        parameter_shapes = {
            name: _tensor_label(parameter)
            for name, parameter in module.named_parameters(recurse=False)
        }
        self._records.append({
            "id": component_id,
            "parent_id": None,
            "name": module_path.rsplit(".", 1)[-1],
            "module_type": type(module).__name__,
            "module_path": module_path,
            "input_ids": input_ids,
            "input_shapes": [_tensor_label(tensor) for tensor in inputs],
            "output_shapes": [_tensor_label(tensor) for tensor in outputs],
            "parameter_names": parameters,
            "parameter_shapes": parameter_shapes,
            "parameter_count": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
        })
        for tensor in outputs:
            self._producers[id(tensor)] = component_id
            if tensor.grad_fn is not None:
                self._gradient_producers[tensor.grad_fn] = component_id

    def _producer_for_tensor(self, tensor: Tensor) -> str | None:
        producer = self._producers.get(id(tensor))
        if producer is not None or tensor.grad_fn is None:
            return producer
        producer = self._gradient_producers.get(tensor.grad_fn)
        if producer is not None:
            return producer
        operation_type = type(tensor.grad_fn).__name__.removesuffix("Backward0")
        invocation = self._operations.get(operation_type, 0) + 1
        self._operations[operation_type] = invocation
        operation_id = f"op:{operation_type}:{invocation}"
        input_ids = _gradient_ancestors(tensor.grad_fn, self._gradient_producers)
        self._records.append({
            "id": operation_id,
            "parent_id": None,
            "name": operation_type,
            "module_type": operation_type,
            "module_path": "",
            "input_ids": list(dict.fromkeys(input_ids)),
            "input_shapes": [],
            "output_shapes": [_tensor_label(tensor)],
            "parameter_names": [],
            "parameter_shapes": {},
            "parameter_count": 0,
        })
        self._producers[id(tensor)] = operation_id
        self._gradient_producers[tensor.grad_fn] = operation_id
        return operation_id

    def write(self, path: Path) -> None:
        assert self._records, "no module execution was recorded"
        path.write_text(json.dumps(self._records, separators=(",", ":")), encoding="utf-8")

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _tensors(value: Any) -> list[Tensor]:
    return [item for item in tree_leaves(value) if isinstance(item, Tensor)]


def _tensor_label(tensor: Tensor) -> str:
    shape = "×".join(str(size) for size in tensor.shape) or "scalar"
    dtype = str(tensor.dtype).removeprefix("torch.")
    return f"{shape} · {dtype}"


def _gradient_ancestors(root: Any, producers: dict[Any, str]) -> list[str]:
    found: list[str] = []
    pending = [root]
    visited: set[Any] = set()
    while pending:
        function = pending.pop()
        if function in visited:
            continue
        visited.add(function)
        producer = producers.get(function)
        if producer is not None:
            found.append(producer)
            continue
        pending.extend(next_function for next_function, _ in function.next_functions if next_function is not None)
    return found
