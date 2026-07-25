import random
from typing import Protocol

from ..config.data import GroupSamplingConfig
from .cuts import CutPlanner
from .index import DatabaseSegmentIndex
from .records import CutRange, EmbeddingGroupPlan, EmbeddingViewPlan
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
        self._validate_pools()

    def plan(
        self,
        batch_index: int,
    ) -> tuple[tuple[EmbeddingGroupPlan, ...], tuple[EmbeddingGroupPlan, ...]]:
        voice_rng = random.Random(
            derive_seed(self.seed, batch_index, "voice-groups")
        )
        speaker_ids = sorted(
            speaker_id
            for speaker_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        )
        global_voice_count = self.grouping.voices_per_batch * self.shard.world_size
        selected_voices = voice_rng.sample(speaker_ids, global_voice_count)
        global_voice_groups = tuple(
            self._voice_group(speaker_id, voice_rng, batch_index)
            for speaker_id in selected_voices
        )
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
        voice_start = self.shard.rank * self.grouping.voices_per_batch
        style_start = self.shard.rank * self.grouping.recordings_per_batch
        voice_groups = global_voice_groups[
            voice_start : voice_start + self.grouping.voices_per_batch
        ]
        style_groups = global_style_groups[
            style_start : style_start + self.grouping.recordings_per_batch
        ]
        return voice_groups, style_groups

    def _validate_pools(self) -> None:
        voices = [
            speaker_id
            for speaker_id, keys in self.index.pools.voice_groups.items()
            if len(keys) >= self.grouping.utterances_per_voice
        ]
        required_voices = self.grouping.voices_per_batch * self.shard.world_size
            )
        required_recordings = (
            self.grouping.recordings_per_batch * self.shard.world_size
        )
            )

    def _voice_group(
        self,
        speaker_id: str,
        rng: random.Random,
        batch_index: int,
    ) -> EmbeddingGroupPlan:
        keys = rng.sample(
            list(self.index.pools.voice_groups[speaker_id]),
            self.grouping.utterances_per_voice,
        )
        views = tuple(
            EmbeddingViewPlan(
                key=key,
                audio=CutRange(self.index.records[key].start, self.index.records[key].end),
                seed=derive_seed(self.seed, batch_index, "voice", speaker_id, key),
                distance_seconds=0,
            )
            for key in keys
        )
        return EmbeddingGroupPlan(speaker_id, views)

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
