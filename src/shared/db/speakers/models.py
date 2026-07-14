import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base
from shared.db.speakers.schemas import EmbeddingRunState


class SpeakerEmbeddingRun(Base):
    __tablename__ = "speaker_embedding_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("speaker_embedding_runs.id", ondelete="CASCADE"), nullable=False, index=True
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
