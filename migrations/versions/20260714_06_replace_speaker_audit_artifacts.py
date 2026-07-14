"""replace speaker audit artifacts with workflow reviews

Revision ID: 20260714_06
Revises: 20260714_05
Create Date: 2026-07-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260714_06"
down_revision: str | None = "20260714_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DELETE FROM speaker_cluster_audits")
    op.drop_constraint(
        "ck_speaker_cluster_audits_completed_integrity",
        "speaker_cluster_audits",
        type_="check",
    )
    op.drop_constraint(
        "speaker_cluster_audits_report_artifact_id_fkey",
        "speaker_cluster_audits",
        type_="foreignkey",
    )
    op.drop_constraint(
        "speaker_cluster_audits_listening_artifact_id_fkey",
        "speaker_cluster_audits",
        type_="foreignkey",
    )
    op.drop_column("speaker_cluster_audits", "report_artifact_id")
    op.drop_column("speaker_cluster_audits", "listening_artifact_id")
    op.add_column(
        "speaker_cluster_audits",
        sa.Column("review_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_speaker_cluster_audits_review_id",
        "speaker_cluster_audits",
        "workflow_reviews",
        ["review_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_speaker_cluster_audits_review_id",
        "speaker_cluster_audits",
        ["review_id"],
    )
    op.create_check_constraint(
        "ck_speaker_cluster_audits_completed_integrity",
        "speaker_cluster_audits",
        "(state = 'open' AND review_id IS NULL AND metrics IS NULL "
        "AND completed_at IS NULL) OR "
        "(state = 'completed' AND review_id IS NOT NULL AND metrics IS NOT NULL "
        "AND completed_at IS NOT NULL)",
    )


def downgrade() -> None:
    op.execute("DELETE FROM speaker_cluster_audits")
    op.drop_constraint(
        "ck_speaker_cluster_audits_completed_integrity",
        "speaker_cluster_audits",
        type_="check",
    )
    op.drop_index(
        "ix_speaker_cluster_audits_review_id",
        table_name="speaker_cluster_audits",
    )
    op.drop_constraint(
        "fk_speaker_cluster_audits_review_id",
        "speaker_cluster_audits",
        type_="foreignkey",
    )
    op.drop_column("speaker_cluster_audits", "review_id")
    op.add_column(
        "speaker_cluster_audits",
        sa.Column("report_artifact_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "speaker_cluster_audits",
        sa.Column(
            "listening_artifact_id", postgresql.UUID(as_uuid=True), nullable=True
        ),
    )
    op.create_foreign_key(
        "speaker_cluster_audits_report_artifact_id_fkey",
        "speaker_cluster_audits",
        "extra_files",
        ["report_artifact_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "speaker_cluster_audits_listening_artifact_id_fkey",
        "speaker_cluster_audits",
        "extra_files",
        ["listening_artifact_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_speaker_cluster_audits_completed_integrity",
        "speaker_cluster_audits",
        "(state = 'open' AND report_artifact_id IS NULL "
        "AND listening_artifact_id IS NULL AND metrics IS NULL "
        "AND completed_at IS NULL) OR "
        "(state = 'completed' AND report_artifact_id IS NOT NULL "
        "AND listening_artifact_id IS NOT NULL AND metrics IS NOT NULL "
        "AND completed_at IS NOT NULL)",
    )
