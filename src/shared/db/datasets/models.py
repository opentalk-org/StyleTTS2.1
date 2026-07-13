import uuid

from sqlalchemy import Column, ForeignKey, Index, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base


dataset_audio_files = Table(
    "dataset_audio_files",
    Base.metadata,
    Column("dataset_id", UUID(as_uuid=True), ForeignKey("datasets.id"), primary_key=True),
    Column("audio_file_id", UUID(as_uuid=True), ForeignKey("audio_files.id"), primary_key=True),
    Index("ix_dataset_audio_files_audio_file_id", "audio_file_id"),
)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    audio_files: Mapped[list["AudioFile"]] = relationship(secondary=dataset_audio_files, back_populates="datasets")
