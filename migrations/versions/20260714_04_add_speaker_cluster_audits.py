"""add speaker cluster audits

Revision ID: 20260714_04
Revises: 20260714_03
Create Date: 2026-07-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_04"
down_revision: str | None = "20260714_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "speaker_cluster_audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("cluster_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("report_artifact_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "listening_artifact_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(state = 'open' AND report_artifact_id IS NULL "
            "AND listening_artifact_id IS NULL AND metrics IS NULL "
            "AND completed_at IS NULL) OR "
            "(state = 'completed' AND report_artifact_id IS NOT NULL "
            "AND listening_artifact_id IS NOT NULL AND metrics IS NOT NULL "
            "AND completed_at IS NOT NULL)",
            name="ck_speaker_cluster_audits_completed_integrity",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_run_id"], ["speaker_clustering_runs.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["report_artifact_id"], ["extra_files.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["listening_artifact_id"], ["extra_files.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "cluster_run_id", "seed", name="uq_speaker_cluster_audits_run_seed"
        ),
    )
    op.create_index(
        "ix_speaker_cluster_audits_cluster_run_id",
        "speaker_cluster_audits",
        ["cluster_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_speaker_cluster_audits_cluster_run_id",
        table_name="speaker_cluster_audits",
    )
    op.drop_table("speaker_cluster_audits")
