"""add speaker clustering runs

Revision ID: 20260714_02
Revises: 20260714_01
Create Date: 2026-07-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_02"
down_revision: str | None = "20260714_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "speaker_clustering_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("embedding_run_id", sa.UUID(), nullable=False),
        sa.Column("expected_count", sa.BigInteger(), nullable=False),
        sa.Column("assignment_count", sa.BigInteger(), nullable=False),
        sa.Column("index_factory", sa.Text(), nullable=False),
        sa.Column("threshold_version", sa.Text(), nullable=False),
        sa.Column("settings", postgresql.JSONB(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("prototype_artifact_id", sa.UUID(), nullable=True),
        sa.Column("index_artifact_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["embedding_run_id"], ["speaker_embedding_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["prototype_artifact_id"], ["extra_files.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["index_artifact_id"], ["extra_files.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_speaker_clustering_runs_embedding_run_id",
        "speaker_clustering_runs",
        ["embedding_run_id"],
    )
    op.create_table(
        "speaker_clustering_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("artifact_id", sa.UUID(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("row_count", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["speaker_clustering_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["artifact_id"], ["extra_files.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id", "role", "ordinal", name="uq_speaker_cluster_artifact_order"
        ),
        sa.UniqueConstraint(
            "run_id", "artifact_id", name="uq_speaker_cluster_artifact_id"
        ),
    )
    op.create_index(
        "ix_speaker_clustering_artifacts_run_id",
        "speaker_clustering_artifacts",
        ["run_id"],
    )
    op.create_table(
        "speaker_cluster_summaries",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("cluster_key", sa.Text(), nullable=False),
        sa.Column("member_count", sa.BigInteger(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("dispersion", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("voice_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"], ["speaker_clustering_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["voice_id"], ["voices.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id", "cluster_key", name="uq_speaker_cluster_summary_key"
        ),
    )
    op.create_index(
        "ix_speaker_cluster_summaries_run_id",
        "speaker_cluster_summaries",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_speaker_cluster_summaries_run_id", table_name="speaker_cluster_summaries"
    )
    op.drop_table("speaker_cluster_summaries")
    op.drop_index(
        "ix_speaker_clustering_artifacts_run_id",
        table_name="speaker_clustering_artifacts",
    )
    op.drop_table("speaker_clustering_artifacts")
    op.drop_index(
        "ix_speaker_clustering_runs_embedding_run_id",
        table_name="speaker_clustering_runs",
    )
    op.drop_table("speaker_clustering_runs")
