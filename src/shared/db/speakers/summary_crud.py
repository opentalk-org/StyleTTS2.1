from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.speakers.models import SpeakerClusterSummary
from shared.db.speakers.schemas import SpeakerClusterStatus
from shared.db.voices import crud as voice_crud
from shared.db.voices.schemas import VoiceCreate


@dataclass(frozen=True)
class ClusterVoicePage:
    voice_ids: dict[str, UUID]
    after: str | None
    created_count: int


def list_cluster_summaries(
    session: Session,
    run_id: UUID,
    status: SpeakerClusterStatus | None = None,
) -> list[SpeakerClusterSummary]:
    statement = select(SpeakerClusterSummary).where(
        SpeakerClusterSummary.run_id == run_id
    )
    if status is not None:
        statement = statement.where(SpeakerClusterSummary.status == status.value)
    return list(
        session.scalars(statement.order_by(SpeakerClusterSummary.cluster_key))
    )


def assign_cluster_summary_voices(
    session: Session,
    run_id: UUID,
    voice_ids: dict[str, UUID],
) -> None:
    if not voice_ids:
        return
    summaries = list(
        session.scalars(
            select(SpeakerClusterSummary)
            .where(
                SpeakerClusterSummary.run_id == run_id,
                SpeakerClusterSummary.cluster_key.in_(voice_ids),
            )
            .with_for_update()
        )
    )
    loaded = {summary.cluster_key: summary for summary in summaries}
    missing = set(voice_ids).difference(loaded)
    if missing:
        raise KeyError(f"speaker cluster summaries not found: {sorted(missing)}")
    for cluster_key, voice_id in voice_ids.items():
        summary = loaded[cluster_key]
        if summary.voice_id is not None and summary.voice_id != voice_id:
            raise ValueError(f"cluster summary {cluster_key} already has another voice")
        summary.voice_id = voice_id
    session.commit()


def reconcile_cluster_summary_voice_page(
    session: Session,
    run_id: UUID,
    after: str | None,
    limit: int,
) -> ClusterVoicePage:
    if limit <= 0:
        raise ValueError("cluster summary page limit must be positive")
    statement = select(SpeakerClusterSummary).where(
        SpeakerClusterSummary.run_id == run_id,
        SpeakerClusterSummary.status == SpeakerClusterStatus.ACCEPTED.value,
    )
    if after is not None:
        statement = statement.where(SpeakerClusterSummary.cluster_key > after)
    summaries = list(
        session.scalars(
            statement.order_by(SpeakerClusterSummary.cluster_key).limit(limit)
        )
    )
    if not summaries:
        return ClusterVoicePage({}, None, 0)
    resolved = {
        summary.cluster_key: summary.voice_id
        for summary in summaries
        if summary.voice_id is not None
    }
    unresolved = [summary for summary in summaries if summary.voice_id is None]
    names = {
        summary.cluster_key: f"speaker-cluster-{run_id}-{summary.cluster_key}"
        for summary in unresolved
    }
    existing = voice_crud.get_voices_by_names(session, list(names.values()))
    missing = [name for name in names.values() if name not in existing]
    created = voice_crud.bulk_create_voices(
        session, [VoiceCreate(name=name) for name in missing]
    )
    by_name = {**existing, **{voice.name: voice for voice in created}}
    assigned = {key: by_name[name].id for key, name in names.items()}
    assign_cluster_summary_voices(session, run_id, assigned)
    return ClusterVoicePage(
        {**resolved, **assigned}, summaries[-1].cluster_key, len(created)
    )
