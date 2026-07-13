import os
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

# Importing every model module registers it on Base.metadata, which Alembic's env.py
# relies on (it imports this module) for autogenerate.
from shared.db.assets import models as asset_models
from shared.db.audio import models as audio_models
from shared.db.datasets import models as dataset_models
from shared.db.initialization import models as initialization_models
from shared.db.jobs import models as job_models
from shared.db.mos import models as mos_models
from shared.db.runners import models as runner_models
from shared.db.settings import models as settings_models
from shared.db.statistics import models as statistics_models
from shared.db.voices import models as voice_models
from shared.db.waveforms import models as waveform_models
from shared.db.workflows import models as workflow_models


DATABASE_URL_ENV = "RUNFLOW_PGBOUNCER_DATABASE_URL"

# migrations/ lives at the repo root (src/shared/db/connection.py -> parents[3]).
# RUNFLOW_ALEMBIC_DIR overrides it for deployments that relocate the source tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS_DIR = Path(os.environ.get("RUNFLOW_ALEMBIC_DIR", str(_REPO_ROOT / "migrations")))


def pgbouncer_engine(database_url: str | None = None) -> Engine:
    url = database_url if database_url is not None else os.environ[DATABASE_URL_ENV]
    return create_engine(url, poolclass=NullPool, future=True)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    # env.py reads this instead of the env var so a caller-supplied URL is honored.
    config.attributes["db_url"] = database_url
    return config


def create_database_schema(database_url: str | None = None) -> None:
    """Bring the database schema up to date by running Alembic migrations to head.

    A fresh database gets the full schema; an up-to-date one is a no-op.
    """
    url = database_url if database_url is not None else os.environ[DATABASE_URL_ENV]
    command.upgrade(_alembic_config(url), "head")


@contextmanager
def database_session(database_url: str | None = None) -> Iterator[Session]:
    engine = pgbouncer_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        with factory() as session:
            try:
                yield session
            except BaseException:
                session.rollback()
                raise
    finally:
        engine.dispose()
