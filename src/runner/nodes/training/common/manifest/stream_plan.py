from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID

from shared.db import database_session
from shared.db.audio import crud as audio_crud

STREAM_PLAN_FILENAME = "stream_plan.json"


@dataclass(frozen=True)
class StreamBucket:
    """One bucket file and the training audio ids that live inside it.

    ``byte_length`` is the on-disk size the unpacked member wavs occupy, used by
    the training-time cache to keep the resident set within its storage budget."""

    bucket_id: UUID
    audio_ids: list[UUID]
    byte_length: int


@dataclass(frozen=True)
class StreamPlan:
    buckets: list[StreamBucket]

    def ordered_audio_ids(self) -> list[UUID]:
        return [audio_id for bucket in self.buckets for audio_id in bucket.audio_ids]


def build_stream_plan(train_audio_ids: list[UUID], speaker_of: dict[UUID, str]) -> StreamPlan:
    """Group training audio by bucket and order buckets to cluster speakers.

    Fetching happens whole-bucket-at-a-time, so members of a bucket are kept
    together. Buckets are then visited grouped by their dominant speaker so the
    cache's resident window holds same-speaker samples for reference selection."""
    with database_session() as session:
        locations = audio_crud.audio_bucket_locations(session, train_audio_ids)
    members: dict[UUID, list[UUID]] = defaultdict(list)
    sizes: dict[UUID, int] = defaultdict(int)
    for location in locations:
        members[location.bucket_file_id].append(location.audio_file_id)
        sizes[location.bucket_file_id] += location.byte_length
    ordered_bucket_ids = sorted(members, key=lambda bucket_id: (_dominant_speaker(members[bucket_id], speaker_of), str(bucket_id)))
    buckets = [
        StreamBucket(
            bucket_id=bucket_id,
            audio_ids=sorted(members[bucket_id], key=lambda audio_id: (speaker_of[audio_id], str(audio_id))),
            byte_length=sizes[bucket_id],
        )
        for bucket_id in ordered_bucket_ids
    ]
    return StreamPlan(buckets=buckets)


def write_stream_plan(path: Path, plan: StreamPlan) -> None:
    payload = {
        "buckets": [
            {
                "bucket_id": str(bucket.bucket_id),
                "audio_ids": [str(audio_id) for audio_id in bucket.audio_ids],
                "byte_length": bucket.byte_length,
            }
            for bucket in plan.buckets
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _dominant_speaker(audio_ids: list[UUID], speaker_of: dict[UUID, str]) -> str:
    counts = Counter(speaker_of[audio_id] for audio_id in audio_ids)
    speaker, _ = counts.most_common(1)[0]
    return speaker
