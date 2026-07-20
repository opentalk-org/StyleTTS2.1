"""add canonical audio annotation columns

Revision ID: 20260720_01
Revises: 20260716_02
Create Date: 2026-07-20 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260720_01"
down_revision: str | None = "20260716_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("audio_files", sa.Column("speaker_id", sa.Text(), nullable=True))
    op.add_column("audio_files", sa.Column("voice_id", sa.UUID(), nullable=True))
    op.add_column("audio_files", sa.Column("accuracy", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("audio_files", "accuracy")
    op.drop_column("audio_files", "voice_id")
    op.drop_column("audio_files", "speaker_id")
