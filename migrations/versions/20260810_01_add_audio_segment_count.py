"""normalize audio segments

Revision ID: 20260810_01
Revises: 20260808_01
Create Date: 2026-08-10 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260810_01"
down_revision: str | None = "20260808_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


INDEX_NAME = "ix_audio_files_segment_count_id"


def upgrade() -> None:
    op.create_table(
        "segments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("audio_file_id", sa.UUID(), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("phon", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("accuracy", sa.Float()),
        sa.Column("speaker_id", sa.Text()),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("position >= 0"),
        sa.CheckConstraint("start_seconds >= 0"),
        sa.CheckConstraint("end_seconds >= start_seconds"),
        sa.ForeignKeyConstraint(["audio_file_id"], ["audio_files.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audio_file_id", "position"),
    )
    op.create_index(
        "ix_segments_audio_file_start",
        "segments",
        ["audio_file_id", "start_seconds", "position"],
    )
    op.create_index(
        "ix_segments_speaker_id",
        "segments",
        ["speaker_id"],
        postgresql_where=sa.text("speaker_id IS NOT NULL"),
    )
    op.create_table(
        "alignments",
        sa.Column("segment_id", sa.BigInteger(), nullable=False),
        sa.Column("data", postgresql.JSONB(none_as_null=False), nullable=False),
        sa.ForeignKeyConstraint(["segment_id"], ["segments.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("segment_id"),
    )
    op.add_column(
        "audio_files",
        sa.Column(
            "segment_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.execute(
        "INSERT INTO segments ("
        "audio_file_id, source_id, position, start_seconds, end_seconds, text, "
        "phon, kind, accuracy, speaker_id, metadata"
        ") SELECT audio.id, COALESCE(segment->>'id', gen_random_uuid()::text), "
        "ordinality::integer - 1, (segment->>'start')::double precision, "
        "(segment->>'end')::double precision, segment->>'text', segment->>'phon', "
        "COALESCE(segment->>'type_', 'segment'), "
        "COALESCE((segment->'annotations'->>'accuracy')::double precision, audio.accuracy), "
        "segment->'annotations'->>'speaker_id', "
        "COALESCE(segment->'annotations'->'metadata', '{}'::jsonb) || "
        "jsonb_build_object('_source', jsonb_build_object("
        "'segment', segment - ARRAY['id', 'start', 'end', 'text', 'phon', "
        "'type_', 'alignment', 'annotations']::text[], "
        "'annotations', COALESCE(segment->'annotations', '{}'::jsonb) - "
        "ARRAY['accuracy', 'speaker_id', 'metadata']::text[])) "
        "FROM audio_files AS audio CROSS JOIN LATERAL "
        "jsonb_array_elements(audio.segments) WITH ORDINALITY AS item(segment, ordinality)"
    )
    op.execute(
        "INSERT INTO alignments (segment_id, data) "
        "SELECT normalized.id, COALESCE("
        "audio.segments->normalized.position->'alignment', 'null'::jsonb) "
        "FROM segments AS normalized "
        "JOIN audio_files AS audio ON audio.id = normalized.audio_file_id"
    )
    op.execute("UPDATE audio_files SET segment_count = jsonb_array_length(segments)")
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
        op.execute(
            f"CREATE INDEX CONCURRENTLY {INDEX_NAME} "
            "ON audio_files (segment_count DESC, id)"
        )
    op.drop_column("audio_files", "segments")
    op.drop_column("audio_files", "accuracy")


def downgrade() -> None:
    op.add_column("audio_files", sa.Column("accuracy", sa.Float()))
    op.add_column(
        "audio_files",
        sa.Column(
            "segments",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.execute(
        "UPDATE audio_files AS audio SET segments = COALESCE(source.payload, '[]'::jsonb) "
        "FROM (SELECT normalized.audio_file_id, jsonb_agg("
        "normalized.metadata->'_source'->'segment' || jsonb_build_object("
        "'id', normalized.source_id, 'start', normalized.start_seconds, "
        "'end', normalized.end_seconds, 'text', normalized.text, "
        "'phon', normalized.phon, 'type_', normalized.kind, "
        "'annotations', normalized.metadata->'_source'->'annotations' || "
        "jsonb_build_object('accuracy', normalized.accuracy, "
        "'speaker_id', normalized.speaker_id, "
        "'metadata', normalized.metadata - '_source'), "
        "'alignment', alignment.data) "
        "ORDER BY normalized.position) AS payload FROM segments AS normalized "
        "JOIN alignments AS alignment ON alignment.segment_id = normalized.id "
        "GROUP BY normalized.audio_file_id) AS source WHERE audio.id = source.audio_file_id"
    )
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
    op.drop_column("audio_files", "segment_count")
    op.drop_table("alignments")
    op.drop_index("ix_segments_speaker_id", table_name="segments")
    op.drop_index("ix_segments_audio_file_start", table_name="segments")
    op.drop_table("segments")
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY {INDEX_NAME} "
            "ON audio_files (jsonb_array_length(segments) DESC, id)"
        )
