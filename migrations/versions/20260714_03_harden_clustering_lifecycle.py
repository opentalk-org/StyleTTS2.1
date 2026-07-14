"""harden clustering lifecycle

Revision ID: 20260714_03
Revises: 20260714_02
Create Date: 2026-07-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_03"
down_revision: str | None = "20260714_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "speaker_clustering_runs",
        sa.Column("outcome_counts", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "speaker_clustering_runs",
        sa.Column("failure_details", postgresql.JSONB(), nullable=True),
    )
    op.drop_constraint(
        "ck_speaker_clustering_artifacts_role",
        "speaker_clustering_artifacts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_speaker_clustering_artifacts_role",
        "speaker_clustering_artifacts",
        "role IN ('candidate', 'assignment', 'prototype', 'index', 'manifest')",
    )
    op.drop_constraint(
        "speaker_clustering_artifacts_artifact_id_fkey",
        "speaker_clustering_artifacts",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "speaker_clustering_artifacts_artifact_id_fkey",
        "speaker_clustering_artifacts",
        "extra_files",
        ["artifact_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint(
        "speaker_clustering_artifacts_artifact_id_fkey",
        "speaker_clustering_artifacts",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "speaker_clustering_artifacts_artifact_id_fkey",
        "speaker_clustering_artifacts",
        "extra_files",
        ["artifact_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint(
        "ck_speaker_clustering_artifacts_role",
        "speaker_clustering_artifacts",
        type_="check",
    )
    op.create_check_constraint(
        "ck_speaker_clustering_artifacts_role",
        "speaker_clustering_artifacts",
        "role IN ('candidate', 'assignment', 'prototype', 'index')",
    )
    op.drop_column("speaker_clustering_runs", "failure_details")
    op.drop_column("speaker_clustering_runs", "outcome_counts")
