"""rename wandb_url setting to mlflow_url

Revision ID: 20260716_02
Revises: 20260716_01
Create Date: 2026-07-16 00:10:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260716_02"
down_revision: str | None = "20260716_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "integration_settings",
        "wandb_url",
        new_column_name="mlflow_url",
        existing_type=sa.Text(),
        server_default="http://localhost:7860",
    )


def downgrade() -> None:
    op.alter_column(
        "integration_settings",
        "mlflow_url",
        new_column_name="wandb_url",
        existing_type=sa.Text(),
        server_default="http://localhost:7860",
    )
