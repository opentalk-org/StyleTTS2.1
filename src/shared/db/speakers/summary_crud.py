from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.speakers.models import SpeakerClusterSummary
from shared.db.speakers.schemas import SpeakerClusterStatus


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
