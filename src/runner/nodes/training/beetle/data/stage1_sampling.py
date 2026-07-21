import random
from dataclasses import dataclass

from .index import DatabaseSegmentIndex
from .records import SegmentKey
from .sampling import DistributedShard, PoolState, derive_seed
from .stage1_records import (
    Stage1PlannedBatch,
    Stage1WindowGeometry,
    Stage1WindowPlan,
)


@dataclass(frozen=True)
class Stage1PlannerState:
    source: PoolState
    pending: tuple[tuple[Stage1WindowPlan, ...], ...]
    batch_index: int

    @property
    def cycle_index(self) -> int:
        return self.source.cycle_index


class _SourcePool:
    def __init__(self, keys: tuple[SegmentKey, ...], seed: int) -> None:
        self.keys = keys
        self.seed = seed
        self.cycle_index = 0
        self.permutation = self._permutation(0)
        self.next_position = 0

    def next(self) -> SegmentKey:
        if self.next_position == len(self.permutation):
            self.cycle_index += 1
            self.permutation = self._permutation(self.cycle_index)
            self.next_position = 0
        key = self.permutation[self.next_position]
        self.next_position += 1
        return key

    def state(self) -> PoolState:
        return PoolState(self.cycle_index, self.permutation, self.next_position)

    def restore(self, state: PoolState) -> None:
        expected = self._permutation(state.cycle_index)
        if state.permutation != expected:
            raise ValueError("Stage 1 source permutation does not match seed")
        if state.next_position < 0 or state.next_position > len(expected):
            raise ValueError("Stage 1 source position is invalid")
        self.cycle_index = state.cycle_index
        self.permutation = state.permutation
        self.next_position = state.next_position

    def _permutation(self, index: int) -> tuple[SegmentKey, ...]:
        values = list(self.keys)
        random.Random(derive_seed(self.seed, "stage-1-windows", index)).shuffle(values)
        return tuple(values)


class Stage1WindowPlanner:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        batch_size: int,
        seed: int,
        shard: DistributedShard,
        geometry: Stage1WindowGeometry,
    ) -> None:
        index.report.require(1, 1.0)
        self.index = index
        self.batch_size = batch_size
        self.shard = shard
        self.geometry = geometry
        self.source = _SourcePool(index.pools.stage1, seed)
        self.pending = tuple(() for _ in range(shard.world_size))
        self.batch_index = 0

    def next_batch(self) -> Stage1PlannedBatch:
        queues = [list(queue) for queue in self.pending]
        while min(map(len, queues)) < self.batch_size:
            key = self.source.next()
            rank = min(
                range(self.shard.world_size),
                key=lambda value: (len(queues[value]), value),
            )
            queues[rank].extend(self.geometry.plans(self.index.records[key]))
        consumed = tuple(tuple(queue[: self.batch_size]) for queue in queues)
        self.pending = tuple(tuple(queue[self.batch_size :]) for queue in queues)
        self.batch_index += 1
        return Stage1PlannedBatch(consumed[self.shard.rank])

    def state_dict(self) -> Stage1PlannerState:
        return Stage1PlannerState(self.source.state(), self.pending, self.batch_index)

    def load_state_dict(self, state: Stage1PlannerState) -> None:
        if len(state.pending) != self.shard.world_size:
            raise ValueError("Stage 1 pending queues do not match world size")
        if state.batch_index < 0:
            raise ValueError("Stage 1 batch index must be non-negative")
        self.source.restore(state.source)
        self.pending = state.pending
        self.batch_index = state.batch_index
