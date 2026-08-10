"""remove audio-level speaker identity

Revision ID: 20260808_01
Revises: 20260806_02
Create Date: 2026-08-08 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260808_01"
down_revision: str | None = "20260806_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_audio_files_speaker_id_id", table_name="audio_files")
    op.drop_column("audio_files", "speaker_id")


def downgrade() -> None:
    op.add_column("audio_files", sa.Column("speaker_id", sa.Text(), nullable=True))
    op.execute(
        "UPDATE audio_files AS audio SET speaker_id = source.speaker_id "
        "FROM ("
        "SELECT DISTINCT ON (audio_file_id) audio_file_id, speaker_id "
        "FROM segments ORDER BY audio_file_id, position"
        ") AS source WHERE source.audio_file_id = audio.id"
    )
    op.create_index(
        "ix_audio_files_speaker_id_id",
        "audio_files",
        ["speaker_id", "id"],
    )
