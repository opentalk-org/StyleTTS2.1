from sqlalchemy.orm import Session

from shared.db.audio.ranges.reader import BulkWavReader
from shared.db.audio.ranges.types import SegmentReadRequest
from shared.db.audio.ranges.wav import WavClip
from shared.db.settings import crud as settings_crud
from shared.storage import S3ObjectStore


def bulk_read_wav_segments(
    session: Session,
    requests: list[SegmentReadRequest],
    worker_count: int,
) -> list[WavClip]:
    store = S3ObjectStore(settings_crud.object_store_config(session))
    reader = BulkWavReader(store, None, worker_count)
    try:
        return reader.read(session, tuple(requests))
    finally:
        reader.close()
