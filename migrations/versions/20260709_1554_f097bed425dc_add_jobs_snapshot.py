"""add jobs.snapshot

Revision ID: f097bed425dc
Revises: 2d2ca95c9fb8
Create Date: 2026-07-09 15:54:27.056189
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'f097bed425dc'
down_revision: str | None = '2d2ca95c9fb8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('jobs', 'snapshot')
