import uuid

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class StorageSettings(Base):
    __tablename__ = "storage_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bucket: Mapped[str] = mapped_column(Text, nullable=False)
    folder: Mapped[str] = mapped_column(Text, nullable=False, default="/")
    endpoint_url: Mapped[str] = mapped_column(Text, nullable=False)
    region_name: Mapped[str] = mapped_column(String(64), nullable=False)
    access_key_id: Mapped[str] = mapped_column(Text, nullable=False)
    secret_access_key: Mapped[str] = mapped_column(Text, nullable=False)


class IntegrationSettings(Base):
    """Credentials for external services (e.g. Hugging Face) used by downloads."""

    __tablename__ = "integration_settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hf_token: Mapped[str] = mapped_column(Text, nullable=False, default="")
    openrouter_token: Mapped[str] = mapped_column(Text, nullable=False, default="")
    aim_url: Mapped[str] = mapped_column(Text, nullable=False, default="http://localhost:43800")
