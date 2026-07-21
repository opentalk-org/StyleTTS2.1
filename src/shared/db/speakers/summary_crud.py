from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.speakers.models import SpeakerClusterSummary
from shared.db.speakers.schemas import SpeakerClusterStatus


@dataclass(frozen=True)
class ClusterSpeakerPage:
    speaker_ids: dict[str, str]
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


def reconcile_cluster_summary_speaker_page(
    session: Session,
    run_id: UUID,
    after: str | None,
    limit: int,
) -> ClusterSpeakerPage:
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
        return ClusterSpeakerPage({}, None, 0)
    speaker_ids = {
        summary.cluster_key: f"speaker-cluster-{run_id}-{summary.cluster_key}"
        for summary in summaries
    }
    return ClusterSpeakerPage(speaker_ids, summaries[-1].cluster_key, 0)
