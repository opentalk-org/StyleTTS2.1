"""add MOS comparisons

Revision ID: 91e06b9c7440
Revises: f097bed425dc
Create Date: 2026-07-10 12:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "91e06b9c7440"
down_revision: str | None = "f097bed425dc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mos_comparisons",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("audio_a_id", sa.UUID(), nullable=False),
        sa.Column("audio_b_id", sa.UUID(), nullable=False),
        sa.Column("preferred_audio_id", sa.UUID(), nullable=False),
        sa.Column("score_a", sa.Float(), nullable=False),
        sa.Column("score_b", sa.Float(), nullable=False),
        sa.Column("previous_score_a", sa.Float(), nullable=True),
        sa.Column("previous_score_b", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("audio_a_id <> audio_b_id", name="ck_mos_comparisons_distinct_audio"),
        sa.CheckConstraint(
            "preferred_audio_id = audio_a_id OR preferred_audio_id = audio_b_id",
            name="ck_mos_comparisons_preferred_member",
        ),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["audio_a_id"], ["audio_files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["audio_b_id"], ["audio_files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["preferred_audio_id"], ["audio_files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_mos_comparisons_dataset_created",
        "mos_comparisons",
        ["dataset_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_mos_comparisons_dataset_created", table_name="mos_comparisons")
    op.drop_table("mos_comparisons")
