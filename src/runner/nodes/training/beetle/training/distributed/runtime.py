from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypeVar

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch import nn
from torch import distributed as dist

from ...config.training import Precision
from ...data import DistributedShard
from ..callbacks import (
    CancellationRequested,
    ProgressEvent,
    TrainingCallbacks,
    TrainingMetric,
)
from ..state import RankState, capture_rng_state, restore_rng_state


ModuleT = TypeVar("ModuleT", bound=nn.Module)
OptimizerT = TypeVar("OptimizerT", bound=torch.optim.Optimizer)


class DistributedCallbacks:
    def __init__(
        self,
        callbacks: TrainingCallbacks,
        runtime: "DistributedRuntime",
    ) -> None:
        self.callbacks = callbacks
        self.runtime = runtime

    def check_cancel(self) -> None:
        cancelled = False
        try:
            self.callbacks.check_cancel()
        except CancellationRequested:
            cancelled = True
        if self.runtime.any_process(cancelled):
            raise CancellationRequested("training cancellation requested")

    def report_progress(self, event: ProgressEvent) -> None:
        if self.runtime.is_main_process:
            self.callbacks.report_progress(event)

    def publish_artifact(self, path: Path, media_type: str) -> None:
        if self.runtime.is_main_process:
            self.callbacks.publish_artifact(path, media_type)


class DistributedRuntime:
    """Process-local view of Accelerate's data-parallel runtime."""

    def __init__(self, precision: Precision, requested_device: torch.device) -> None:
        self.precision = precision
        ddp = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            cpu=requested_device.type == "cpu",
            mixed_precision="no",
            kwargs_handlers=[ddp],
        )
        if self.device.type != requested_device.type:
            raise ValueError("Accelerate device type does not match runtime.device")

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def rank(self) -> int:
        return self.accelerator.process_index

    @property
    def world_size(self) -> int:
        return self.accelerator.num_processes

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def shard(self) -> DistributedShard:
        return DistributedShard(self.rank, self.world_size)

    def prepare_module(self, module: ModuleT) -> ModuleT:
        return self.accelerator.prepare_model(module)

    def prepare_optimizer(self, optimizer: OptimizerT) -> OptimizerT:
        return self.accelerator.prepare_optimizer(optimizer)

    def unwrap(self, module: ModuleT) -> ModuleT:
        return self.accelerator.unwrap_model(module)

    def autocast(self) -> AbstractContextManager[None]:
        if self.precision is Precision.FLOAT32:
            return torch.autocast(self.device.type, enabled=False)
        dtype = (
            torch.bfloat16
            if self.precision is Precision.BFLOAT16
            else torch.float16
        )
        return torch.autocast(self.device.type, dtype=dtype)

    def reduce_metrics(
        self,
        metrics: tuple[TrainingMetric, ...],
    ) -> tuple[TrainingMetric, ...]:
        if not metrics:
            return ()
        values = torch.tensor(
            tuple(metric.value for metric in metrics),
            dtype=torch.float64,
            device=self.device,
        )
        reduced = self.accelerator.reduce(values, reduction="mean").cpu().tolist()
        return tuple(
            TrainingMetric(metric.name, value)
            for metric, value in zip(metrics, reduced, strict=True)
        )

    def gather_objects(self, value: object) -> tuple[object, ...]:
        if self.world_size == 1:
            return (value,)
        values: list[object | None] = [None] * self.world_size
        dist.all_gather_object(values, value)
        if any(item is None for item in values):
            raise RuntimeError("distributed object gather returned an empty rank")
        return tuple(values)

    def broadcast_object(self, value: object) -> object:
        values = [value]
        if self.world_size > 1:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def any_process(self, value: bool) -> bool:
        flag = torch.tensor(int(value), dtype=torch.int32, device=self.device)
        reduced = self.accelerator.reduce(flag, reduction="sum")
        return bool(reduced.item())

    def wait_for_everyone(self) -> None:
        self.accelerator.wait_for_everyone()

    def capture_rank_state(self) -> RankState:
        return RankState(capture_rng_state())

    def restore_rank_state(self, state: RankState) -> None:
        restore_rng_state(state.rng)
