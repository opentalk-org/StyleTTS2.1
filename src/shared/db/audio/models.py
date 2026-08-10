import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.assets.models import BucketFile
from shared.db.base import Base


class AudioFile(Base):
    __tablename__ = "audio_files"
    __table_args__ = (
        Index("ix_audio_files_updated_at_id", text("updated_at DESC"), text("id")),
        Index("ix_audio_files_name_id", text("name"), text("id")),
        Index("ix_audio_files_duration_id", text("duration DESC"), text("id")),
        Index("ix_audio_files_segment_count_id", text("segment_count DESC"), text("id")),
        Index(
            "ix_audio_files_split_operation",
            text("(metadata ->> 'source_audio_id')"),
            text("(metadata ->> 'split_operation_id')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    bucket_file_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("bucket_files.id"), index=True)
    byte_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    byte_length: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    segment_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    language: Mapped[str | None] = mapped_column(Text)
    style_prompt: Mapped[str | None] = mapped_column(Text)
    voice_prompt: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    virtual: Mapped[bool] = mapped_column(Boolean, nullable=False)
    storage_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="packed")
    storage_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    bucket_file: Mapped[BucketFile | None] = relationship(back_populates="audio_files", lazy="joined")
    datasets: Mapped[list["Dataset"]] = relationship(secondary="dataset_audio_files", back_populates="audio_files")
    segment_rows: Mapped[list["AudioSegment"]] = relationship(
        back_populates="audio_file",
        cascade="all, delete-orphan",
        order_by="AudioSegment.position",
        lazy="selectin",
    )

    @property
    def segments(self) -> list[dict[str, Any]]:
        return [segment.as_payload() for segment in self.segment_rows]

class AudioSegment(Base):
    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("audio_file_id", "position"),
        CheckConstraint("position >= 0"),
        CheckConstraint("start_seconds >= 0"),
        CheckConstraint("end_seconds >= start_seconds"),
        Index("ix_segments_audio_file_start", "audio_file_id", "start_seconds", "position"),
        Index("ix_segments_speaker_id", "speaker_id", postgresql_where=text("speaker_id IS NOT NULL")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audio_files.id", ondelete="RESTRICT"), nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    phon: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float)
    speaker_id: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False)
    audio_file: Mapped[AudioFile] = relationship(back_populates="segment_rows")
    alignment: Mapped["Alignment"] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
        lazy="joined",
        uselist=False,
    )

    def as_payload(self) -> dict[str, Any]:
        source = self.metadata_["_source"]
        annotations = {
            **source["annotations"],
            "accuracy": self.accuracy,
            "speaker_id": self.speaker_id,
            "metadata": {
                key: value
                for key, value in self.metadata_.items()
                if key != "_source"
            },
        }
        return {
            **source["segment"],
            "id": self.source_id,
            "start": self.start_seconds,
            "end": self.end_seconds,
            "text": self.text,
            "phon": self.phon,
            "type_": self.kind,
            "annotations": annotations,
            "alignment": self.alignment.data,
        }

    def as_source_payload(self) -> dict[str, Any]:
        source = self.metadata_["_source"]
        return {
            **source["segment"],
            "id": self.source_id,
            "start": self.start_seconds,
            "end": self.end_seconds,
            "text": self.text,
            "phon": self.phon,
            "type_": self.kind,
            "annotations": {
                **source["annotations"],
                "metadata": {
                    key: value
                    for key, value in self.metadata_.items()
                    if key != "_source"
                },
            },
            "alignment": self.alignment.data,
        }


class Alignment(Base):
    __tablename__ = "alignments"

    segment_id: Mapped[int] = mapped_column(
        ForeignKey("segments.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    data: Mapped[Any] = mapped_column(JSONB(none_as_null=False), nullable=False)
    segment: Mapped[AudioSegment] = relationship(back_populates="alignment")
