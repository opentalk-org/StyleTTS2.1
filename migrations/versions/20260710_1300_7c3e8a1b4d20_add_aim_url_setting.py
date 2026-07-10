"""add Aim URL setting

Revision ID: 7c3e8a1b4d20
Revises: 2d91f62f4e8a
Create Date: 2026-07-10 13:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "7c3e8a1b4d20"
down_revision: str | None = "2d91f62f4e8a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "integration_settings",
        sa.Column("aim_url", sa.Text(), nullable=False, server_default="http://localhost:43800"),
    )


def downgrade() -> None:
    op.drop_column("integration_settings", "aim_url")
