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
    mid_sentence: PoolState | None
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

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("distributed world size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("distributed rank must be within world size")


class _PermutationPool:
    def __init__(self, keys: tuple[SegmentKey, ...], seed: int, label: str) -> None:
        if not keys:
            raise ValueError(f"sampling pool is empty: {label}")
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
        if state.permutation != expected:
            raise ValueError(f"{self.label} permutation does not match seed/cycle")
        if state.next_position < 0 or state.next_position > len(expected):
            raise ValueError(f"{self.label} next_position is invalid")
        self.cycle_index = state.cycle_index
        self.permutation = state.permutation
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
        sentence_probability: float,
        seed: int,
        maximum_seconds: float,
        grouping: GroupSamplingConfig,
        shard: DistributedShard,
    ) -> None:
        index.report.require(sentence_probability)
        self.index = index
        self.batch_size = batch_size
        self.sentence_probability = sentence_probability
        self.seed = seed
        self.shard = shard
        self.cut_planner = CutPlanner(index, maximum_seconds)
        self.sentence = _PermutationPool(
            index.pools.segments,
            seed,
            "training-sentence",
        )
        self.mid_sentence = (
            None
            if sentence_probability == 1
            else _PermutationPool(
                index.pools.mid_sentence,
                seed,
                "training-mid",
            )
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
        if batch_count <= 0:
            raise ValueError("window batch count must be positive")
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
            sentence = random.Random(sample_seed).random() < self.sentence_probability
            pool = self.sentence if sentence else self._require_mid_sentence_pool()
            key, position = pool.next()
            cut_seed = derive_seed(sample_seed, pool.cycle_index, position, key)
            plan = (
                self.cut_planner.plan_sentence(key, cut_seed)
                if sentence
                else self.cut_planner.plan_mid_sentence(key, cut_seed)
            )
            plans.append(plan)
        plans.sort(key=lambda plan: (plan.target.duration, plan.key))
        global_batches = [
            tuple(plans[start : start + global_batch_size])
            for start in range(0, len(plans), global_batch_size)
        ]
        random.Random(
            derive_seed(self.seed, start_batch_index, "window-order")
        ).shuffle(global_batches)
        group_batches = tuple(
            self.embedding_groups.plan(planned_batch_index)
            for planned_batch_index in range(
                start_batch_index,
                start_batch_index + batch_count,
            )
        )
        local_start = self.shard.rank * self.batch_size
        local_end = local_start + self.batch_size
        self.pending_batches = tuple(
            PlannedBatch(global_batch[local_start:local_end], voice_groups, style_groups)
            for global_batch, (voice_groups, style_groups) in zip(
                global_batches,
                group_batches,
                strict=True,
            )
        )
        self.planning_batch_index += batch_count
        mid_cycle = 0 if self.mid_sentence is None else self.mid_sentence.cycle_index
        self.last_cycle_index = max(self.sentence.cycle_index, mid_cycle)

    def state_dict(self) -> PlannerState:
        return PlannerState(
            sentence=self.sentence.state(),
            mid_sentence=(
                None if self.mid_sentence is None else self.mid_sentence.state()
            ),
            batch_index=self.batch_index,
            planning_batch_index=self.planning_batch_index,
            pending_batches=self.pending_batches,
            last_cycle_index=self.last_cycle_index,
        )

    def load_state_dict(self, state: PlannerState) -> None:
        if state.batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if state.planning_batch_index < state.batch_index:
            raise ValueError("planning batch index precedes consumed batches")
        self.sentence.restore(state.sentence)
        if self.mid_sentence is None:
            if state.mid_sentence is not None:
                raise ValueError("sentence-only planner state contains a mid-sentence pool")
        else:
            if state.mid_sentence is None:
                raise ValueError("mid-sentence planner state is missing its pool")
            self.mid_sentence.restore(state.mid_sentence)
        self.batch_index = state.batch_index
        self.planning_batch_index = state.planning_batch_index
        self.pending_batches = state.pending_batches
        self.last_cycle_index = state.last_cycle_index

    def _require_mid_sentence_pool(self) -> _PermutationPool:
        if self.mid_sentence is None:
            raise RuntimeError("sentence-only planner selected a mid-sentence sample")
        return self.mid_sentence
