"""add jobs.attempts for runner-crash retry cap

Revision ID: 20260715_01
Revises: 20260714_06
Create Date: 2026-07-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260715_01"
down_revision: str | None = "20260714_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("jobs", "attempts")
