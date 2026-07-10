"""add audio_files.style_prompt and voice_prompt

Revision ID: a1b2c3d4e5f6
Revises: 7c3e8a1b4d20
Create Date: 2026-07-10 13:30:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "7c3e8a1b4d20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("audio_files", sa.Column("style_prompt", sa.Text(), nullable=True))
    op.add_column("audio_files", sa.Column("voice_prompt", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("audio_files", "voice_prompt")
    op.drop_column("audio_files", "style_prompt")
