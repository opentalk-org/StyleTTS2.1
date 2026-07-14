"""add speaker embedding runs

Revision ID: 20260714_01
Revises: e6f7a8b9c0d1
Create Date: 2026-07-14 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_01"
down_revision: str | None = "e6f7a8b9c0d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "speaker_embedding_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=False),
        sa.Column("expected_count", sa.BigInteger(), nullable=False),
        sa.Column("stored_count", sa.BigInteger(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("model_revision", sa.Text(), nullable=False),
        sa.Column("preprocessing_version", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("failure_details", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sealed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_speaker_embedding_runs_dataset_id", "speaker_embedding_runs", ["dataset_id"]
    )
    op.create_table(
        "speaker_embedding_shards",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("artifact_id", sa.UUID(), nullable=False),
        sa.Column("row_count", sa.BigInteger(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("model_revision", sa.Text(), nullable=False),
        sa.Column("preprocessing_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["extra_files.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["run_id"], ["speaker_embedding_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id", "artifact_id", name="uq_speaker_embedding_shards_run_artifact"
        ),
    )
    op.create_index(
        "ix_speaker_embedding_shards_run_id", "speaker_embedding_shards", ["run_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_speaker_embedding_shards_run_id", table_name="speaker_embedding_shards")
    op.drop_table("speaker_embedding_shards")
    op.drop_index("ix_speaker_embedding_runs_dataset_id", table_name="speaker_embedding_runs")
    op.drop_table("speaker_embedding_runs")
