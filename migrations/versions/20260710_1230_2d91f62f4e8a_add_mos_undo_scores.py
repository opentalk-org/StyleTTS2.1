"""add MOS undo scores

Revision ID: 2d91f62f4e8a
Revises: 91e06b9c7440
Create Date: 2026-07-10 12:30:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "2d91f62f4e8a"
down_revision: str | None = "91e06b9c7440"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("mos_comparisons", sa.Column("previous_score_a", sa.Float(), nullable=True))
    op.add_column("mos_comparisons", sa.Column("previous_score_b", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("mos_comparisons", "previous_score_b")
    op.drop_column("mos_comparisons", "previous_score_a")
