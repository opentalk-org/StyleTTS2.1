import random
from typing import Protocol

from ..config.data import GroupSamplingConfig
from .cuts import CutPlanner
from .index import DatabaseSegmentIndex
from .records import (
    CutRange,
    EmbeddingGroupPlan,
    EmbeddingViewPlan,
    PlannedExample,
)
from .seeding import derive_seed


class Shard(Protocol):
    rank: int
    world_size: int


class EmbeddingGroupPlanner:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        grouping: GroupSamplingConfig,
        shard: Shard,
        cut_planner: CutPlanner,
        seed: int,
    ) -> None:
        self.index = index
        self.grouping = grouping
        self.shard = shard
        self.cut_planner = cut_planner
        self.seed = seed

    def plan(
        self,
        examples: tuple[PlannedExample, ...],
        batch_index: int,
    ) -> tuple[
        tuple[EmbeddingGroupPlan, ...],
        tuple[EmbeddingGroupPlan, ...],
        tuple[int, ...],
        int,
    ]:
        voice_rng = random.Random(
            derive_seed(self.seed, batch_index, "voice-groups")
        )
        eligible_speakers = sorted(
            speaker_id
            for speaker_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        )
        global_voice_count = self.grouping.voices_per_batch * self.shard.world_size
        selected_auxiliary = voice_rng.sample(
            eligible_speakers,
            global_voice_count,
        )
        global_auxiliary_groups = tuple(
            self._voice_group(speaker_id, voice_rng, batch_index)
            for speaker_id in selected_auxiliary
        )
        voice_start = self.shard.rank * self.grouping.voices_per_batch
        voice_end = voice_start + self.grouping.voices_per_batch
        auxiliary_groups = global_auxiliary_groups[voice_start:voice_end]
        voice_groups = list(auxiliary_groups)
        voice_condition_indices = []
        for example in examples:
            condition_index = self._voice_condition_index(example, voice_groups)
            if condition_index is None:
                view = self._voice_condition_view(
                    example,
                    voice_rng,
                    batch_index,
                )
                speaker_id = self.index.records[example.key].speaker_id
                assert speaker_id is not None
                condition_index = sum(
                    len(group.views) for group in voice_groups
                )
                voice_groups.append(EmbeddingGroupPlan(speaker_id, (view,)))
            voice_condition_indices.append(condition_index)
        style_rng = random.Random(
            derive_seed(self.seed, batch_index, "style-groups")
        )
        recording_ids = sorted(self.index.pools.recording_groups, key=str)
        global_recording_count = (
            self.grouping.recordings_per_batch * self.shard.world_size
        )
        selected_recordings = style_rng.sample(recording_ids, global_recording_count)
        global_style_groups = tuple(
            self._style_group(recording_id, style_rng, batch_index)
            for recording_id in selected_recordings
        )
        style_start = self.shard.rank * self.grouping.recordings_per_batch
        style_groups = global_style_groups[
            style_start : style_start + self.grouping.recordings_per_batch
        ]
        return (
            tuple(voice_groups),
            style_groups,
            tuple(voice_condition_indices),
            sum(len(group.views) for group in auxiliary_groups),
        )

    def _voice_group(
        self,
        speaker_id: str,
        rng: random.Random,
        batch_index: int,
    ) -> EmbeddingGroupPlan:
        candidates = list(self.index.pools.voice_groups[speaker_id])
        rng.shuffle(candidates)
        keys = []
        recording_ids = set()
        for key in candidates:
            if key.audio_file_id not in recording_ids:
                keys.append(key)
                recording_ids.add(key.audio_file_id)
            if len(keys) == self.grouping.utterances_per_voice:
                break
        for key in candidates:
            if len(keys) == self.grouping.utterances_per_voice:
                break
            if key not in keys:
                keys.append(key)
        while len(keys) < self.grouping.utterances_per_voice:
            keys.append(rng.choice(candidates))
        views = tuple(
            EmbeddingViewPlan(
                key=key,
                audio=CutRange(self.index.records[key].start, self.index.records[key].end),
                seed=derive_seed(
                    self.seed,
                    batch_index,
                    "voice",
                    speaker_id,
                    view_index,
                    key,
                ),
                distance_seconds=0,
            )
            for view_index, key in enumerate(keys)
        )
        return EmbeddingGroupPlan(speaker_id, views)

    def _voice_condition_index(
        self,
        example: PlannedExample,
        groups: list[EmbeddingGroupPlan],
    ) -> int | None:
        speaker_id = self.index.records[example.key].speaker_id
        flat_index = 0
        fallback = None
        for group in groups:
            for view in group.views:
                if group.group_id == speaker_id:
                    fallback = flat_index
                    if view.key.audio_file_id != example.key.audio_file_id:
                        return flat_index
                flat_index += 1
        distinct_wav_exists = any(
            key.audio_file_id != example.key.audio_file_id
            for key in self.index.pools.voice_groups[speaker_id]
        )
        return None if distinct_wav_exists else fallback

    def _voice_condition_view(
        self,
        example: PlannedExample,
        rng: random.Random,
        batch_index: int,
    ) -> EmbeddingViewPlan:
        target = self.index.records[example.key]
        assert target.speaker_id is not None
        candidates = tuple(
            key
            for key in self.index.pools.voice_groups[target.speaker_id]
            if key.audio_file_id != example.key.audio_file_id
        )
        key = rng.choice(candidates) if candidates else example.key
        item = self.index.records[key]
        return EmbeddingViewPlan(
            key,
            CutRange(item.start, item.end),
            derive_seed(
                self.seed,
                batch_index,
                "voice-condition",
                example.key,
                key,
            ),
            0,
        )

    def _style_group(
        self,
        recording_id: object,
        rng: random.Random,
        batch_index: int,
    ) -> EmbeddingGroupPlan:
        keys = self.index.pools.recording_groups[recording_id]
        selected = [rng.choice(keys) for _ in range(self.grouping.cuts_per_recording)]
        views = []
        first_center = None
        for view_index, key in enumerate(selected):
            item = self.index.records[key]
            seed = derive_seed(self.seed, batch_index, "style", recording_id, view_index)
            audio = (
                self.cut_planner.plan_mid_sentence(key, seed).target
                if item.mid_sentence_eligible
                else CutRange(item.start, item.end)
            )
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
