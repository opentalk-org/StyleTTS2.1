from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID as PythonUUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    desired_state: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    target_runner_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    claimed_runner_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    graph_request: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Last per-node run snapshot (statuses/errors/counters), so a run reopened after a
    # restart still shows which node failed rather than a blank graph.
    snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    review_count: ClassVar[int] = 0


class NodeLog(Base):
    __tablename__ = "node_logs"
    __table_args__ = (UniqueConstraint("run_id", "node_id", name="uq_node_logs_run_node"),)

    id: Mapped[PythonUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))


class RunNodeState(Base):
    __tablename__ = "run_node_states"
    __table_args__ = (UniqueConstraint("run_id", "node_id", name="uq_run_node_states_run_node"),)

    id: Mapped[PythonUUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[str] = mapped_column(ForeignKey("jobs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    node_id: Mapped[str] = mapped_column(Text, nullable=False)
    desired_loaded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    observed_loaded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
