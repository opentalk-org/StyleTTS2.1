"""add external audio storage

Revision ID: c4e5f6a7b8c9
Revises: a1b2c3d4e5f6
Create Date: 2026-07-13 12:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c4e5f6a7b8c9"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "audio_files",
        sa.Column("storage_kind", sa.String(length=16), nullable=False, server_default="packed"),
    )
    op.add_column(
        "audio_files",
        sa.Column("storage_ref", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.alter_column("audio_files", "bucket_file_id", existing_type=sa.UUID(), nullable=True)
    op.create_check_constraint(
        "ck_audio_files_storage",
        "audio_files",
        "(storage_kind = 'packed' AND bucket_file_id IS NOT NULL AND storage_ref IS NULL) OR "
        "(storage_kind = 'external' AND bucket_file_id IS NULL AND storage_ref IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_audio_files_storage", "audio_files", type_="check")
    op.alter_column("audio_files", "bucket_file_id", existing_type=sa.UUID(), nullable=False)
    op.drop_column("audio_files", "storage_ref")
    op.drop_column("audio_files", "storage_kind")
