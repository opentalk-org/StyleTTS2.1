from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from runner.nodes.models import Audio


@dataclass(frozen=True)
class EmbeddingBatchIdentity:
    dataset_id: UUID
    source_segment_count: int


def validate_embedding_batch(
    audios: list[Audio],
    run_dataset_id: UUID | None,
    run_source_segment_count: int | None,
) -> EmbeddingBatchIdentity:
    assert (run_dataset_id is None) is (run_source_segment_count is None), (
        "embedding run identity is incomplete"
    )
    identities = [_audio_identity(audio, index) for index, audio in enumerate(audios)]
    expected = (
        identities[0]
        if run_dataset_id is None
        else EmbeddingBatchIdentity(run_dataset_id, run_source_segment_count)
    )
    for index, identity in enumerate(identities):
        if identity.dataset_id != expected.dataset_id:
            raise ValueError(
                f"audio item {index} dataset_id {identity.dataset_id} does not match "
                f"embedding run dataset_id {expected.dataset_id}"
            )
        if identity.source_segment_count != expected.source_segment_count:
            raise ValueError(
                f"audio item {index} source_segment_count {identity.source_segment_count} "
                f"does not match embedding run source_segment_count "
                f"{expected.source_segment_count}"
            )
    return expected


def bounded_audio_groups(audios: list[Audio], maximum_seconds: float) -> list[list[Audio]]:
    groups: list[list[Audio]] = []
    current: list[Audio] = []
    current_seconds = 0.0
    for audio in audios:
        if audio.duration > maximum_seconds:
            continue
        if current and current_seconds + audio.duration > maximum_seconds:
            groups.append(current)
            current = []
            current_seconds = 0.0
        current.append(audio)
        current_seconds += audio.duration
    if current:
        groups.append(current)
    return groups


def _audio_identity(audio: Audio, index: int) -> EmbeddingBatchIdentity:
    try:
        raw_dataset_id = audio.metadata["dataset_id"]
    except KeyError as error:
        raise ValueError(f"audio item {index} is missing dataset_id metadata") from error
    try:
        dataset_id = UUID(str(raw_dataset_id))
    except ValueError as error:
        raise ValueError(
            f"audio item {index} has invalid dataset_id metadata: {raw_dataset_id!r}"
        ) from error
    try:
        raw_count = audio.metadata["source_segment_count"]
    except KeyError as error:
        raise ValueError(
            f"audio item {index} is missing source_segment_count metadata"
        ) from error
    try:
        source_segment_count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"audio item {index} has invalid source_segment_count metadata: {raw_count!r}"
        ) from error
    return EmbeddingBatchIdentity(dataset_id, source_segment_count)
