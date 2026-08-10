"""restrict alignment deletes to the segment lifecycle

Revision ID: 20260810_02
Revises: 20260810_01
Create Date: 2026-08-10 00:10:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260810_02"
down_revision: str | None = "20260810_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CONSTRAINT_NAME = "alignments_segment_id_fkey"


def upgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "alignments", type_="foreignkey")
    op.create_foreign_key(
        CONSTRAINT_NAME,
        "alignments",
        "segments",
        ["segment_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "alignments", type_="foreignkey")
    op.create_foreign_key(
        CONSTRAINT_NAME,
        "alignments",
        "segments",
        ["segment_id"],
        ["id"],
        ondelete="CASCADE",
    )
