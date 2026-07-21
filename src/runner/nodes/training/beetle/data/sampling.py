import hashlib
import random
from dataclasses import dataclass

from ..config.data import GroupSamplingConfig
from .cuts import CutPlanner
from .index import DatabaseSegmentIndex
from .records import (
    CutRange,
    EmbeddingGroupPlan,
    EmbeddingViewPlan,
    PlannedBatch,
    SegmentKey,
)


def derive_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False) & ((1 << 63) - 1)


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

    @property
    def cycle_index(self) -> int:
        mid_sentence_cycle = (
            0 if self.mid_sentence is None else self.mid_sentence.cycle_index
        )
        return max(self.sentence.cycle_index, mid_sentence_cycle)


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
        stage: int,
        batch_size: int,
        sentence_probability: float,
        seed: int,
        maximum_seconds: float,
        grouping: GroupSamplingConfig,
        shard: DistributedShard,
    ) -> None:
        index.report.require(stage, sentence_probability)
        self.index = index
        self.stage = stage
        self.batch_size = batch_size
        self.sentence_probability = sentence_probability
        self.seed = seed
        self.grouping = grouping
        self.shard = shard
        self.cut_planner = CutPlanner(index, maximum_seconds)
        self.sentence = _PermutationPool(index.pools.for_stage(stage), seed, f"stage-{stage}-sentence")
        self.mid_sentence = (
            None
            if sentence_probability == 1
            else _PermutationPool(
                index.pools.mid_sentence,
                seed,
                f"stage-{stage}-mid",
            )
        )
        self.batch_index = 0
        self._validate_embedding_pools()

    def next_batch(self) -> PlannedBatch:
        plans = []
        global_batch_size = self.batch_size * self.shard.world_size
        for sample_index in range(global_batch_size):
            sample_seed = derive_seed(
                self.seed,
                self.stage,
                self.batch_index,
                sample_index,
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
        start = self.shard.rank * self.batch_size
        examples = tuple(plans[start : start + self.batch_size])
        voice_groups, style_groups = self._embedding_groups()
        self.batch_index += 1
        return PlannedBatch(examples, voice_groups, style_groups)

    def state_dict(self) -> PlannerState:
        return PlannerState(
            sentence=self.sentence.state(),
            mid_sentence=(
                None if self.mid_sentence is None else self.mid_sentence.state()
            ),
            batch_index=self.batch_index,
        )

    def load_state_dict(self, state: PlannerState) -> None:
        if state.batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        self.sentence.restore(state.sentence)
        if self.mid_sentence is None:
            if state.mid_sentence is not None:
                raise ValueError("sentence-only planner state contains a mid-sentence pool")
        else:
            if state.mid_sentence is None:
                raise ValueError("mid-sentence planner state is missing its pool")
            self.mid_sentence.restore(state.mid_sentence)
        self.batch_index = state.batch_index

    def _require_mid_sentence_pool(self) -> _PermutationPool:
        if self.mid_sentence is None:
            raise RuntimeError("sentence-only planner selected a mid-sentence sample")
        return self.mid_sentence

    def _validate_embedding_pools(self) -> None:
        if self.stage == 1:
            return
        voices = [
            voice_id
            for voice_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        ]
        required_voices = (
            self.grouping.voices_per_batch * self.shard.world_size
        )
        if len(voices) < required_voices:
            raise ValueError(
                "voice GE2E sampling requires "
                f"{required_voices} voices with "
                f"{self.grouping.utterances_per_voice} utterances; found {len(voices)}"
            )
        required_recordings = (
            self.grouping.recordings_per_batch * self.shard.world_size
        )
        if len(self.index.pools.recording_groups) < required_recordings:
            raise ValueError(
                "style GE2E sampling requires "
                f"{required_recordings} recordings; found "
                f"{len(self.index.pools.recording_groups)}"
            )

    def _embedding_groups(
        self,
    ) -> tuple[tuple[EmbeddingGroupPlan, ...], tuple[EmbeddingGroupPlan, ...]]:
        if self.stage == 1:
            return (), ()
        voice_seed = derive_seed(self.seed, self.stage, self.batch_index, "voice-groups")
        voice_rng = random.Random(voice_seed)
        voice_ids = sorted(
            voice_id
            for voice_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        )
        global_voice_count = (
            self.grouping.voices_per_batch * self.shard.world_size
        )
        selected_voices = voice_rng.sample(voice_ids, global_voice_count)
        global_voice_groups = tuple(
            self._voice_group(voice_id, voice_rng)
            for voice_id in selected_voices
        )
        style_seed = derive_seed(self.seed, self.stage, self.batch_index, "style-groups")
        style_rng = random.Random(style_seed)
        recording_ids = sorted(self.index.pools.recording_groups, key=str)
        global_recording_count = (
            self.grouping.recordings_per_batch * self.shard.world_size
        )
        selected_recordings = style_rng.sample(
            recording_ids,
            global_recording_count,
        )
        global_style_groups = tuple(
            self._style_group(recording_id, style_rng)
            for recording_id in selected_recordings
        )
        voice_start = self.shard.rank * self.grouping.voices_per_batch
        style_start = self.shard.rank * self.grouping.recordings_per_batch
        voice_groups = global_voice_groups[
            voice_start : voice_start + self.grouping.voices_per_batch
        ]
        style_groups = global_style_groups[
            style_start : style_start + self.grouping.recordings_per_batch
        ]
        return voice_groups, style_groups

    def _voice_group(self, voice_id: str, rng: random.Random) -> EmbeddingGroupPlan:
        keys = rng.sample(
            list(self.index.pools.voice_groups[voice_id]),
            self.grouping.utterances_per_voice,
        )
        views = tuple(
            EmbeddingViewPlan(
                key=key,
                audio=CutRange(self.index.records[key].start, self.index.records[key].end),
                seed=derive_seed(self.seed, self.batch_index, "voice", voice_id, key),
                distance_seconds=0,
            )
            for key in keys
        )
        return EmbeddingGroupPlan(voice_id, views)

    def _style_group(self, recording_id: object, rng: random.Random) -> EmbeddingGroupPlan:
        keys = self.index.pools.recording_groups[recording_id]
        selected = [rng.choice(keys) for _ in range(self.grouping.cuts_per_recording)]
        views = []
        first_center = None
        for view_index, key in enumerate(selected):
            item = self.index.records[key]
            seed = derive_seed(self.seed, self.batch_index, "style", recording_id, view_index)
            if item.mid_sentence_eligible:
                audio = self.cut_planner.plan_mid_sentence(key, seed).target
            else:
                audio = CutRange(item.start, item.end)
            center = (audio.start + audio.end) / 2
            first_center = center if first_center is None else first_center
            views.append(
                EmbeddingViewPlan(
                    key=key,
                    audio=audio,
                    seed=seed,
                    distance_seconds=abs(center - first_center),
                )
            )
        return EmbeddingGroupPlan(str(recording_id), tuple(views))
