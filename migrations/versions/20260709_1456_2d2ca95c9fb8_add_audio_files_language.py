"""add audio_files.language

Revision ID: 2d2ca95c9fb8
Revises: fbcd07e2d59d
Create Date: 2026-07-09 14:56:45.645021
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '2d2ca95c9fb8'
down_revision: str | None = 'fbcd07e2d59d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('audio_files', sa.Column('language', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('audio_files', 'language')
