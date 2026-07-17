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
    mid_sentence: PoolState
    batch_index: int


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
        grouping: GroupSamplingConfig,
    ) -> None:
        index.report.require(stage, sentence_probability)
        self.index = index
        self.stage = stage
        self.batch_size = batch_size
        self.sentence_probability = sentence_probability
        self.seed = seed
        self.grouping = grouping
        self.cut_planner = CutPlanner(index, 1.0, 45.0)
        self.sentence = _PermutationPool(index.pools.for_stage(stage), seed, f"stage-{stage}-sentence")
        self.mid_sentence = _PermutationPool(index.pools.mid_sentence, seed, f"stage-{stage}-mid")
        self.batch_index = 0
        self._validate_embedding_pools()

    def next_batch(self) -> PlannedBatch:
        plans = []
        for sample_index in range(self.batch_size):
            sample_seed = derive_seed(
                self.seed,
                self.stage,
                self.batch_index,
                sample_index,
            )
            sentence = random.Random(sample_seed).random() < self.sentence_probability
            pool = self.sentence if sentence else self.mid_sentence
            key, position = pool.next()
            cut_seed = derive_seed(sample_seed, pool.cycle_index, position, key)
            plan = (
                self.cut_planner.plan_sentence(key, cut_seed)
                if sentence
                else self.cut_planner.plan_mid_sentence(key, cut_seed)
            )
            plans.append(plan)
        voice_groups, style_groups = self._embedding_groups()
        self.batch_index += 1
        return PlannedBatch(tuple(plans), voice_groups, style_groups)

    def state_dict(self) -> PlannerState:
        return PlannerState(
            sentence=self.sentence.state(),
            mid_sentence=self.mid_sentence.state(),
            batch_index=self.batch_index,
        )

    def load_state_dict(self, state: PlannerState) -> None:
        if state.batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        self.sentence.restore(state.sentence)
        self.mid_sentence.restore(state.mid_sentence)
        self.batch_index = state.batch_index

    def _validate_embedding_pools(self) -> None:
        if self.stage == 1:
            return
        voices = [
            voice_id
            for voice_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        ]
        if len(voices) < self.grouping.voices_per_batch:
            raise ValueError(
                "voice GE2E sampling requires "
                f"{self.grouping.voices_per_batch} voices with "
                f"{self.grouping.utterances_per_voice} utterances; found {len(voices)}"
            )
        if len(self.index.pools.recording_groups) < self.grouping.recordings_per_batch:
            raise ValueError(
                "style GE2E sampling requires "
                f"{self.grouping.recordings_per_batch} recordings; found "
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
        selected_voices = voice_rng.sample(voice_ids, self.grouping.voices_per_batch)
        voice_groups = tuple(
            self._voice_group(voice_id, voice_rng)
            for voice_id in selected_voices
        )
        style_seed = derive_seed(self.seed, self.stage, self.batch_index, "style-groups")
        style_rng = random.Random(style_seed)
        recording_ids = sorted(self.index.pools.recording_groups, key=str)
        selected_recordings = style_rng.sample(
            recording_ids,
            self.grouping.recordings_per_batch,
        )
        style_groups = tuple(
            self._style_group(recording_id, style_rng)
            for recording_id in selected_recordings
        )
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
