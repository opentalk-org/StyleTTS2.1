"""postgres run coordination

Revision ID: d5e6f7a8b9c0
Revises: c4e5f6a7b8c9
Create Date: 2026-07-13 14:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d5e6f7a8b9c0"
down_revision: str | None = "c4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("desired_state", sa.Text(), nullable=False, server_default="running"))
    op.add_column("jobs", sa.Column("target_runner_id", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("claimed_runner_id", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_jobs_target_runner_id", "jobs", ["target_runner_id"])
    op.create_index("ix_jobs_claimed_runner_id", "jobs", ["claimed_runner_id"])

    op.add_column("runners", sa.Column("process_id", sa.Integer(), nullable=True))
    op.add_column("runners", sa.Column("active_run_ids", postgresql.JSONB(), nullable=False, server_default="[]"))
    op.add_column("runners", sa.Column("capabilities", postgresql.JSONB(), nullable=False, server_default="{}"))
    op.add_column("runners", sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_runners_last_seen_at", "runners", ["last_seen_at"])

    op.create_table(
        "run_node_states",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("node_id", sa.Text(), nullable=False),
        sa.Column("desired_loaded", sa.Boolean(), nullable=False),
        sa.Column("observed_loaded", sa.Boolean(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["jobs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "node_id", name="uq_run_node_states_run_node"),
    )
    op.create_index("ix_run_node_states_run_id", "run_node_states", ["run_id"])


def downgrade() -> None:
    op.drop_table("run_node_states")
    op.drop_index("ix_runners_last_seen_at", table_name="runners")
    op.drop_column("runners", "last_seen_at")
    op.drop_column("runners", "capabilities")
    op.drop_column("runners", "active_run_ids")
    op.drop_column("runners", "process_id")
    op.drop_index("ix_jobs_claimed_runner_id", table_name="jobs")
    op.drop_index("ix_jobs_target_runner_id", table_name="jobs")
    op.drop_column("jobs", "lease_expires_at")
    op.drop_column("jobs", "claimed_runner_id")
    op.drop_column("jobs", "target_runner_id")
    op.drop_column("jobs", "desired_state")
