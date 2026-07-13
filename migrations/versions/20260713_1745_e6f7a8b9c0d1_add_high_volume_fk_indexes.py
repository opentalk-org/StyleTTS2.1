"""add high-volume foreign-key indexes

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
Create Date: 2026-07-13 17:45:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "e6f7a8b9c0d1"
down_revision: str | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_dataset_audio_files_audio_file_id", "dataset_audio_files", ["audio_file_id"])
    op.create_index("ix_audio_files_bucket_file_id", "audio_files", ["bucket_file_id"])
    op.create_index("ix_audio_waveforms_pack_id", "audio_waveforms", ["pack_id"])
    op.create_index("ix_mos_comparisons_audio_a_id", "mos_comparisons", ["audio_a_id"])
    op.create_index("ix_mos_comparisons_audio_b_id", "mos_comparisons", ["audio_b_id"])
    op.create_index("ix_mos_comparisons_preferred_audio_id", "mos_comparisons", ["preferred_audio_id"])
    op.execute(
        "CREATE INDEX ix_audio_files_split_operation "
        "ON audio_files ((metadata ->> 'source_audio_id'), "
        "(metadata ->> 'split_operation_id'))"
    )


def downgrade() -> None:
    op.drop_index("ix_audio_files_split_operation", table_name="audio_files")
    op.drop_index("ix_mos_comparisons_preferred_audio_id", table_name="mos_comparisons")
    op.drop_index("ix_mos_comparisons_audio_b_id", table_name="mos_comparisons")
    op.drop_index("ix_mos_comparisons_audio_a_id", table_name="mos_comparisons")
    op.drop_index("ix_audio_waveforms_pack_id", table_name="audio_waveforms")
    op.drop_index("ix_audio_files_bucket_file_id", table_name="audio_files")
    op.drop_index("ix_dataset_audio_files_audio_file_id", table_name="dataset_audio_files")
