import logging

from shared.db.audio.maintenance import purge_orphaned_audio_pack_batch
from shared.db.connection import database_session
from shared.db.settings import crud as settings_crud

LOGGER = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    total = 0
    with database_session() as session:
        store = settings_crud.object_store(session)
        store.test_connection()
        while paths := purge_orphaned_audio_pack_batch(session, store):
            total += len(paths)
            LOGGER.info("deleted orphan batch=%d total=%d", len(paths), total)
    LOGGER.info("complete: deleted_orphans=%d", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
