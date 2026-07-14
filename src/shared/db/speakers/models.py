import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base
from shared.db.speakers.schemas import ClusteringRunState, EmbeddingRunState


class SpeakerEmbeddingRun(Base):
    __tablename__ = "speaker_embedding_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    expected_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    stored_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    model_revision: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EmbeddingRunState.OPEN.value
    )
    failure_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    sealed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    shards: Mapped[list["SpeakerEmbeddingShard"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class SpeakerEmbeddingShard(Base):
    __tablename__ = "speaker_embedding_shards"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "artifact_id", name="uq_speaker_embedding_shards_run_artifact"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("speaker_embedding_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extra_files.id", ondelete="RESTRICT"), nullable=False
    )
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    model_revision: Mapped[str] = mapped_column(Text, nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    run: Mapped[SpeakerEmbeddingRun] = relationship(back_populates="shards")


class SpeakerClusteringRun(Base):
    __tablename__ = "speaker_clustering_runs"
    __table_args__ = (
        UniqueConstraint("run_key", name="uq_speaker_clustering_runs_run_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_key: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("speaker_embedding_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    expected_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    assignment_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    index_factory: Mapped[str] = mapped_column(Text, nullable=False)
    threshold_version: Mapped[str] = mapped_column(Text, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ClusteringRunState.OPEN.value
    )
    outcome_counts: Mapped[dict[str, int] | None] = mapped_column(JSONB, nullable=True)
    failure_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    prototype_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extra_files.id", ondelete="RESTRICT"), nullable=True
    )
    index_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extra_files.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class SpeakerClusteringArtifact(Base):
    __tablename__ = "speaker_clustering_artifacts"
    __table_args__ = (
        CheckConstraint(
            "role IN ('candidate', 'assignment', 'prototype', 'index', 'manifest')",
            name="ck_speaker_clustering_artifacts_role",
        ),
        UniqueConstraint(
            "run_id", "role", "ordinal", name="uq_speaker_cluster_artifact_order"
        ),
        UniqueConstraint(
            "run_id", "artifact_id", name="uq_speaker_cluster_artifact_id"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("speaker_clustering_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("extra_files.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int] = mapped_column(BigInteger, nullable=False)


class SpeakerClusterSummary(Base):
    __tablename__ = "speaker_cluster_summaries"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "cluster_key", name="uq_speaker_cluster_summary_key"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("speaker_clustering_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cluster_key: Mapped[str] = mapped_column(Text, nullable=False)
    member_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    dispersion: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    voice_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("voices.id", ondelete="SET NULL"), nullable=True
    )
