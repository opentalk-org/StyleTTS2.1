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
        self._containers: dict[str, dict[str, Any]] = {}
        self._handles = []
        visited: set[int] = set()
        for root_name, root in modules.items():
            for relative_name, module in root.named_modules():
                path = root_name if relative_name == "" else f"{root_name}.{relative_name}"
                if len(tuple(module.children())) > 0:
                    self._containers[path] = _container_record(path, module)
                    continue
                if id(module) in visited:
                    continue
                visited.add(id(module))
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
            "parent_id": _parent_path(module_path),
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
        self._place_operations()
        records = self._records + self._reached_containers()
        path.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")

    def _place_operations(self) -> None:
        """Ops carry no module path, so they take the container of the module that fed them.

        An op that also consumes a tensor from outside that container (a residual add, a
        branch merge) is lifted one level, which is where the merge actually happens.
        """
        parents = {record["id"]: record["parent_id"] for record in self._records}
        for record in self._records:
            if record["module_path"] != "" or len(record["input_ids"]) == 0:
                continue
            container = parents[record["input_ids"][0]]
            outside = any(
                not _contains(container, parents[input_id])
                for input_id in record["input_ids"][1:]
            )
            if outside and container is not None:
                container = _parent_path(container) or container
            record["parent_id"] = container
            parents[record["id"]] = container

    def _reached_containers(self) -> list[dict[str, Any]]:
        """Only containers on the path to an executed module belong in the graph."""
        reached: set[str] = set()
        for record in self._records:
            path = record["parent_id"]
            while path is not None and path not in reached:
                reached.add(path)
                path = _parent_path(path)
        return [self._containers[path] for path in sorted(reached)]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _contains(container: str | None, module_path: str | None) -> bool:
    if container is None:
        return True
    if module_path is None:
        return False
    return module_path == container or module_path.startswith(f"{container}.")


def _parent_path(module_path: str) -> str | None:
    parent = module_path.rsplit(".", 1)[0]
    return None if parent == module_path else parent


def _container_record(module_path: str, module: nn.Module) -> dict[str, Any]:
    return {
        "id": module_path,
        "parent_id": _parent_path(module_path),
        "name": module_path.rsplit(".", 1)[-1],
        "module_type": type(module).__name__,
        "module_path": module_path,
        "parameter_names": [],
        "parameter_shapes": {},
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
    }


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
