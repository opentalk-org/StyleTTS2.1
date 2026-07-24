from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from appdirs import user_cache_dir

from shared.db.audio.ranges import (
    AudioFileCache,
    BulkWavReader,
    SegmentReadRequest,
)
from shared.db.audio.ranges.wav import WavClip
from shared.db.audio import crud as audio_crud
from shared.db.connection import database_session
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore

from .index import DatabaseSegmentIndex
from .records import (
    EmbeddingGroupPlan,
    EmbeddingViewPlan,
    PlannedBatch,
    PlannedExample,
    SegmentKey,
)


class SegmentBulkLoader(Protocol):
    def load(
        self,
        audio_file_ids: tuple[UUID, ...],
    ) -> dict[UUID, list[dict[str, Any]]]: ...


class ClipBulkLoader(Protocol):
    def load(self, requests: tuple[SegmentReadRequest, ...]) -> list[WavClip]: ...

    def close(self) -> None: ...


class SharedSegmentBulkLoader:
    def load(
        self,
        audio_file_ids: tuple[UUID, ...],
    ) -> dict[UUID, list[dict[str, Any]]]:
        with database_session() as session:
            return audio_crud.list_audio_segments_bulk(session, audio_file_ids)


class SharedClipBulkLoader:
    def __init__(self, cache_bytes: int, fetch_workers: int) -> None:
        with database_session() as session:
            store = S3ObjectStore(settings_crud.object_store_config(session))
        cache_root = Path(user_cache_dir("runflow")) / "audio"
        self.reader = BulkWavReader(
            store,
            AudioFileCache(cache_root, cache_bytes),
            fetch_workers,
        )

    def load(self, requests: tuple[SegmentReadRequest, ...]) -> list[WavClip]:
        with database_session() as session:
            return self.reader.read(session, requests)

    def close(self) -> None:
        self.reader.close()


@dataclass(frozen=True)
class FetchedExample:
    plan: PlannedExample
    text: str
    phonemes: str
    target_clip: WavClip
    style_prompt: str | None
    voice_prompt: str | None
    speaker_id: str | None
    language: str | None


@dataclass(frozen=True)
class FetchedEmbeddingView:
    plan: EmbeddingViewPlan
    clip: WavClip


@dataclass(frozen=True)
class FetchedEmbeddingGroup:
    group_id: str
    views: tuple[FetchedEmbeddingView, ...]


@dataclass(frozen=True)
class FetchedBatch:
    examples: tuple[FetchedExample, ...]
    voice_groups: tuple[FetchedEmbeddingGroup, ...]
    style_groups: tuple[FetchedEmbeddingGroup, ...]


class DatabaseBatchSource:
    def __init__(
        self,
        index: DatabaseSegmentIndex,
        segments: SegmentBulkLoader,
        clips: ClipBulkLoader,
    ) -> None:
        self.index = index
        self.segments = segments
        self.clips = clips

    @classmethod
    def from_database(
        cls,
        index: DatabaseSegmentIndex,
        cache_bytes: int,
        fetch_workers: int,
    ) -> "DatabaseBatchSource":
        return cls(
            index,
            SharedSegmentBulkLoader(),
            SharedClipBulkLoader(cache_bytes, fetch_workers),
        )

    def close(self) -> None:
        self.clips.close()

    def fetch(self, planned: PlannedBatch) -> FetchedBatch:
        keys = _batch_keys(planned)
        audio_ids = tuple(sorted({key.audio_file_id for key in keys}, key=str))
        payloads = self.segments.load(audio_ids)
        current = {key: self._resolve_segment(key, payloads) for key in keys}
        requests = _batch_requests(planned)
        clips = self.clips.load(requests)
        if len(clips) != len(requests):
            raise ValueError("bulk WAV loader returned the wrong clip count")
        clip_map = dict(zip(requests, clips, strict=True))
        examples = tuple(
            self._fetched_example(plan, current, clip_map)
            for plan in planned.examples
        )
        voice_groups = tuple(
            _fetched_group(group, clip_map) for group in planned.voice_groups
        )
        style_groups = tuple(
            _fetched_group(group, clip_map) for group in planned.style_groups
        )
        return FetchedBatch(examples, voice_groups, style_groups)

    @staticmethod
    def _resolve_segment(
        key: SegmentKey,
        payloads: dict[UUID, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if key.audio_file_id not in payloads:
            raise KeyError(f"audio file disappeared during prefetch: {key.audio_file_id}")
        segments = payloads[key.audio_file_id]
        if key.segment_index >= len(segments):
            raise ValueError(f"segment-index drift for {key}")
        segment = segments[key.segment_index]
        if str(segment["id"]) != key.segment_id:
            raise ValueError(
                f"segment-ID drift for {key}: database now contains {segment['id']}"
            )
        return segment

    def _fetched_example(
        self,
        plan: PlannedExample,
        current: dict[SegmentKey, dict[str, Any]],
        clips: dict[SegmentReadRequest, WavClip],
    ) -> FetchedExample:
        target = current[plan.key]
        text, phonemes = _target_text(target, plan)
        item = self.index.records[plan.key]
        return FetchedExample(
            plan=plan,
            text=text,
            phonemes=phonemes,
            target_clip=clips[_request(plan.key, plan.target.start, plan.target.end)],
            style_prompt=item.style_prompt,
            voice_prompt=item.voice_prompt,
            speaker_id=item.speaker_id,
            language=item.language,
        )


def _batch_keys(planned: PlannedBatch) -> tuple[SegmentKey, ...]:
    keys = []
    for plan in planned.examples:
        keys.append(plan.key)
    for group in (*planned.voice_groups, *planned.style_groups):
        keys.extend(view.key for view in group.views)
    return tuple(dict.fromkeys(keys))


def _batch_requests(planned: PlannedBatch) -> tuple[SegmentReadRequest, ...]:
    requests = []
    for plan in planned.examples:
        requests.append(_request(plan.key, plan.target.start, plan.target.end))
    for group in (*planned.voice_groups, *planned.style_groups):
        requests.extend(_request(view.key, view.audio.start, view.audio.end) for view in group.views)
    return tuple(dict.fromkeys(requests))


def _request(key: SegmentKey, start: float, end: float) -> SegmentReadRequest:
    return SegmentReadRequest(key.audio_file_id, start, end)


def _target_text(segment: dict[str, Any], plan: PlannedExample) -> tuple[str, str]:
    if plan.sentence:
        return str(segment["text"]), str(segment["phon"])
    alignment = segment["alignment"]
    words = [str(item["word"]) for item in alignment[plan.target_word_start:plan.target_word_end]]
    phonemes = str(segment["phon"]).split()[plan.target_word_start:plan.target_word_end]
    return " ".join(words), " ".join(phonemes)


def _fetched_group(
    group: EmbeddingGroupPlan,
    clips: dict[SegmentReadRequest, WavClip],
) -> FetchedEmbeddingGroup:
    views = tuple(
        FetchedEmbeddingView(
            view,
            clips[_request(view.key, view.audio.start, view.audio.end)],
        )
        for view in group.views
    )
    return FetchedEmbeddingGroup(group.group_id, views)
