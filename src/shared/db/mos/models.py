import uuid
from datetime import UTC, datetime

from sqlalchemy import REAL, CheckConstraint, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class MosComparison(Base):
    __tablename__ = "mos_comparisons"
    __table_args__ = (
        CheckConstraint("audio_a_id <> audio_b_id", name="ck_mos_comparisons_distinct_audio"),
        CheckConstraint(
            "preferred_audio_id = audio_a_id OR preferred_audio_id = audio_b_id",
            name="ck_mos_comparisons_preferred_member",
        ),
        Index("ix_mos_comparisons_dataset_created", "dataset_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    audio_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    audio_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    preferred_audio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("audio_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score_a: Mapped[float] = mapped_column(REAL, nullable=False)
    score_b: Mapped[float] = mapped_column(REAL, nullable=False)
    previous_score_a: Mapped[float | None] = mapped_column(REAL, nullable=True)
    previous_score_b: Mapped[float | None] = mapped_column(REAL, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
