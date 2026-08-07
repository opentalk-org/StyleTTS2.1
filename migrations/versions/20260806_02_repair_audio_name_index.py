"""repair audio name index

Revision ID: 20260806_02
Revises: 20260806_01
Create Date: 2026-08-06 00:10:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260806_02"
down_revision: str | None = "20260806_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


INDEX_NAME = "ix_audio_files_name_id"


def upgrade() -> None:
    valid = op.get_bind().execute(
        sa.text(
            "SELECT i.indisvalid FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname = :name"
        ),
        {"name": INDEX_NAME},
    ).scalar_one_or_none()
    if valid:
        return
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(
            f"CREATE INDEX CONCURRENTLY {INDEX_NAME} "
            "ON audio_files (name, id)"
        )


def downgrade() -> None:
    pass
