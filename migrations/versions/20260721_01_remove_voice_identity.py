"""remove duplicate voice identity

Revision ID: 20260721_01
Revises: 20260720_01
Create Date: 2026-07-21 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260721_01"
down_revision: str | None = "20260720_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("speaker_cluster_summaries", "voice_id")
    op.drop_column("audio_files", "voice_id")
    op.drop_table("voices")


def downgrade() -> None:
    op.create_table(
        "voices",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.add_column("audio_files", sa.Column("voice_id", sa.UUID(), nullable=True))
    op.add_column(
        "speaker_cluster_summaries",
        sa.Column("voice_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_speaker_cluster_summaries_voice_id_voices",
        "speaker_cluster_summaries",
        "voices",
        ["voice_id"],
        ["id"],
        ondelete="SET NULL",
    )
