import uuid
from datetime import UTC, datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base


class WaveformPack(Base):
    __tablename__ = "waveform_packs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    used_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sealed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    waveforms: Mapped[list["AudioWaveform"]] = relationship(back_populates="pack")


class AudioWaveform(Base):
    __tablename__ = "audio_waveforms"

    audio_file_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audio_files.id"), primary_key=True)
    pack_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("waveform_packs.id"), nullable=False, index=True)
    byte_offset: Mapped[int] = mapped_column(BigInteger, nullable=False)
    byte_length: Mapped[int] = mapped_column(BigInteger, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    points_per_second: Mapped[int] = mapped_column(Integer, nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    format_version: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    pack: Mapped[WaveformPack] = relationship(back_populates="waveforms", lazy="joined")
