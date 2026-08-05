import random
from dataclasses import dataclass

from ..config.data import GroupSamplingConfig
from .cuts import CutPlanner
from .index import DatabaseSegmentIndex
from .embedding_sampling import EmbeddingGroupPlanner
from .records import PlannedBatch, SegmentKey
from .seeding import derive_seed


@dataclass(frozen=True)
class PoolState:
    cycle_index: int
    permutation: tuple[SegmentKey, ...]
    next_position: int


@dataclass(frozen=True)
class PlannerState:
    sentence: PoolState
    batch_index: int
    planning_batch_index: int
    pending_batches: tuple[PlannedBatch, ...]
    last_cycle_index: int

    @property
    def cycle_index(self) -> int:
        return self.last_cycle_index


@dataclass(frozen=True)
class PlannedWindowBatch:
    batch: PlannedBatch
    state_after: PlannerState


@dataclass(frozen=True)
class DistributedShard:
    rank: int
    world_size: int

class _PermutationPool:
    def __init__(self, keys: tuple[SegmentKey, ...], seed: int, label: str) -> None:
        self.keys = keys
        self.seed = seed
        self.label = label
        self.cycle_index = 0
        self.permutation = self._permutation(0)
        self.next_position = 0

    def next(self) -> tuple[SegmentKey, int]:
        if self.next_position == len(self.permutation):
            self.cycle_index += 1
            self.permutation = self._permutation(self.cycle_index)
            self.next_position = 0
        key = self.permutation[self.next_position]
        position = self.next_position
        self.next_position += 1
        return key, position

    def state(self) -> PoolState:
        return PoolState(self.cycle_index, self.permutation, self.next_position)

    def restore(self, state: PoolState) -> None:
        expected = self._permutation(state.cycle_index)
        self.cycle_index = state.cycle_index
        self.permutation = expected
        self.next_position = state.next_position

    def _permutation(self, cycle_index: int) -> tuple[SegmentKey, ...]:
        values = list(self.keys)
        random.Random(derive_seed(self.seed, self.label, cycle_index)).shuffle(values)
        return tuple(values)


class ContinuousBatchPlanner:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        batch_size: int,
        seed: int,
        maximum_seconds: float,
        grouping: GroupSamplingConfig,
        shard: DistributedShard,
    ) -> None:
        index.report.require()
        self.index = index
        self.batch_size = batch_size
        self.seed = seed
        self.shard = shard
        self.cut_planner = CutPlanner(index, maximum_seconds)
        self.sentence = _PermutationPool(
            index.pools.segments,
            seed,
            "training-sentence",
        )
        self.batch_index = 0
        self.planning_batch_index = 0
        self.pending_batches: tuple[PlannedBatch, ...] = ()
        self.last_cycle_index = 0
        self.embedding_groups = EmbeddingGroupPlanner(
            index, grouping, shard, self.cut_planner, seed
        )

    def next_batch(self) -> PlannedBatch:
        return self.next_window(1)[0].batch

    def next_window(self, batch_count: int) -> tuple[PlannedWindowBatch, ...]:
        if not self.pending_batches:
            self._plan_window(batch_count)
        take = min(batch_count, len(self.pending_batches))
        output = []
        for _ in range(take):
            batch = self.pending_batches[0]
            self.pending_batches = self.pending_batches[1:]
            self.batch_index += 1
            output.append(PlannedWindowBatch(batch, self.state_dict()))
        return tuple(output)

    def _plan_window(self, batch_count: int) -> None:
        plans = []
        global_batch_size = self.batch_size * self.shard.world_size
        global_window_size = global_batch_size * batch_count
        start_batch_index = self.planning_batch_index
        for sample_index in range(global_window_size):
            planned_batch_index = start_batch_index + sample_index // global_batch_size
            batch_sample_index = sample_index % global_batch_size
            sample_seed = derive_seed(
                self.seed,
                planned_batch_index,
                batch_sample_index,
            )
            key, position = self.sentence.next()
            cut_seed = derive_seed(
                sample_seed,
                self.sentence.cycle_index,
                position,
                key,
            )
            plan = self.cut_planner.plan_sentence(key, cut_seed)
            plans.append(plan)
        plans.sort(key=lambda plan: (plan.target.duration, plan.key))
        global_batches = [
            tuple(plans[start : start + global_batch_size])
            for start in range(0, len(plans), global_batch_size)
        ]
        random.Random(
            derive_seed(self.seed, start_batch_index, "window-order")
        ).shuffle(global_batches)
        local_start = self.shard.rank * self.batch_size
        local_end = local_start + self.batch_size
        pending_batches = []
        for offset, global_batch in enumerate(global_batches):
            local_batch = global_batch[local_start:local_end]
            (
                voice_groups,
                style_groups,
                voice_condition_indices,
                voice_auxiliary_view_count,
            ) = self.embedding_groups.plan(
                local_batch,
                start_batch_index + offset,
            )
            pending_batches.append(
                PlannedBatch(
                    local_batch,
                    voice_groups,
                    style_groups,
                    voice_condition_indices,
                    voice_auxiliary_view_count,
                )
            )
        self.pending_batches = tuple(pending_batches)
        self.planning_batch_index += batch_count
        self.last_cycle_index = self.sentence.cycle_index

    def state_dict(self) -> PlannerState:
        return PlannerState(
            sentence=self.sentence.state(),
            batch_index=self.batch_index,
            planning_batch_index=self.planning_batch_index,
            pending_batches=self.pending_batches,
            last_cycle_index=self.last_cycle_index,
        )

    def load_state_dict(self, state: PlannerState) -> None:
        self.sentence.restore(state.sentence)
        self.batch_index = state.batch_index
        self.planning_batch_index = state.planning_batch_index
        self.pending_batches = state.pending_batches
        self.last_cycle_index = state.last_cycle_index
