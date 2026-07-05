import uuid

from sqlalchemy import BigInteger, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class Runner(Base):
    __tablename__ = "runners"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    hostname: Mapped[str] = mapped_column(Text, nullable=False)
    port: Mapped[int] = mapped_column(BigInteger, nullable=False)
    gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
