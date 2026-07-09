"""Alembic environment.

The target metadata is ``shared.db.base.Base.metadata``; importing
``shared.db.connection`` registers every model module against it, so autogenerate
sees the full schema. The database URL is resolved (in order) from an ``-x db_url=``
command-line override, then ``RUNFLOW_PGBOUNCER_DATABASE_URL`` -- never hard-coded.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# Importing connection populates Base.metadata with every model.
from shared.db.base import Base
import shared.db.connection  # noqa: F401  (side effect: registers all models)

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    # Priority: -x db_url=... (CLI) > cfg.attributes["db_url"] (programmatic, set by
    # shared.db.connection) > RUNFLOW_PGBOUNCER_DATABASE_URL env var.
    override = context.get_x_argument(as_dictionary=True).get("db_url")
    url = override or config.attributes.get("db_url") or os.environ.get("RUNFLOW_PGBOUNCER_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "No database URL: set RUNFLOW_PGBOUNCER_DATABASE_URL or pass -x db_url=..."
        )
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # NullPool + a fresh engine mirrors shared.db.connection.pgbouncer_engine, which is
    # what the app uses against PgBouncer (transaction pooling).
    engine = create_engine(_database_url(), poolclass=NullPool, future=True)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
