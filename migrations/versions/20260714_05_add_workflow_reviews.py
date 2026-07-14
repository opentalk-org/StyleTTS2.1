"""add workflow reviews

Revision ID: 20260714_05
Revises: 20260714_04
Create Date: 2026-07-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_05"
down_revision: str | None = "20260714_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workflow_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("producer_run_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("source_key", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("continuation", postgresql.JSONB(), nullable=True),
        sa.Column("continuation_run_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["producer_run_id"], ["jobs.run_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kind", "source_key", name="uq_workflow_reviews_identity"
        ),
    )
    op.create_index(
        "ix_workflow_reviews_producer_run_id",
        "workflow_reviews",
        ["producer_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workflow_reviews_producer_run_id", table_name="workflow_reviews"
    )
    op.drop_table("workflow_reviews")
