from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.db.assets.models import BucketFile
from shared.db.audio.pack_store import ObjectStore
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore


def purge_orphaned_audio_packs(
    session: Session,
    store: ObjectStore | None = None,
) -> list[str]:
    statement = select(BucketFile).where(~BucketFile.audio_files.any()).with_for_update()
    packs = list(session.execute(statement).scalars().all())
    paths = [pack.path for pack in packs]
    for pack in packs:
        session.delete(pack)
    session.commit()
    resolved_store = store or S3ObjectStore(settings_crud.object_store_config(session))
    for path in paths:
        resolved_store.delete(path)
    return paths
