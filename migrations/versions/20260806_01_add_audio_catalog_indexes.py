"""add audio catalog indexes

Revision ID: 20260806_01
Revises: 20260721_01
Create Date: 2026-08-06 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260806_01"
down_revision: str | None = "20260721_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


INDEXES = (
    ("ix_audio_files_updated_at_id", "updated_at DESC, id"),
    ("ix_audio_files_name_id", "name, id"),
    ("ix_audio_files_duration_id", "duration DESC, id"),
    ("ix_audio_files_speaker_id_id", "speaker_id, id"),
    (
        "ix_audio_files_segment_count_id",
        "jsonb_array_length(segments) DESC, id",
    ),
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        for name, columns in INDEXES:
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} "
                f"ON audio_files ({columns})"
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _ in reversed(INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
