import os
from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from shared.db.assets import models as asset_models
from shared.db.audio import models as audio_models
from shared.db.base import Base
from shared.db.datasets import models as dataset_models
from shared.db.initialization import models as initialization_models
from shared.db.jobs import models as job_models
from shared.db.runners import models as runner_models
from shared.db.settings import models as settings_models
from shared.db.voices import models as voice_models
from shared.db.workflows import models as workflow_models


DATABASE_URL_ENV = "RUNFLOW_PGBOUNCER_DATABASE_URL"


def pgbouncer_engine(database_url: str | None = None) -> Engine:
    url = database_url if database_url is not None else os.environ[DATABASE_URL_ENV]
    return create_engine(url, poolclass=NullPool, future=True)


def create_database_schema(database_url: str | None = None) -> None:
    engine = pgbouncer_engine(database_url)
    try:
        Base.metadata.create_all(engine)
        _ensure_audio_updated_at(engine)
        _ensure_checkpoint_job_id(engine)
    finally:
        engine.dispose()


def _ensure_audio_updated_at(engine: Engine) -> None:
    statement = text("ALTER TABLE audio_files ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()")
    with engine.begin() as connection:
        connection.execute(statement)


def _ensure_checkpoint_job_id(engine: Engine) -> None:
    statement = text("ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS job_id TEXT NULL")
    with engine.begin() as connection:
        connection.execute(statement)


@contextmanager
def database_session(database_url: str | None = None) -> Iterator[Session]:
    engine = pgbouncer_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        with factory() as session:
            yield session
    finally:
        engine.dispose()
