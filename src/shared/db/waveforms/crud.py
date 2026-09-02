import uuid
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shared.db.assets.clickhouse import (
    BucketFileRecord,
    BucketKind,
    create_bucket_files,
    get_bucket_file,
)
from shared.db.assets.crud import delete_unreferenced_bucket_files
from shared.db.settings import crud as settings_crud
from shared.db.waveforms.clickhouse import (
    AudioWaveformRecord,
    delete_waveforms,
    get_waveform,
    get_waveforms,
    replace_waveform as publish_waveform,
)
from shared.db.waveforms.codec import (
    decode_peaks,
    downsample,
    encode_peaks,
    waveform_from_wav,
)
from shared.db.waveforms.schemas import WaveformInput, WaveformRead
from shared.storage import ObjectRange


@dataclass(frozen=True)
class WaveformPackConfig:
    path_prefix: str = "waveform-packs"


def replace_waveform(
    session: Session,
    audio_file_id: uuid.UUID,
    duration: float,
    payload: WaveformInput,
    config: WaveformPackConfig = WaveformPackConfig(),
) -> AudioWaveformRecord:
    data = encode_peaks(payload.peaks)
    pack_id = uuid.uuid4()
    path = f"{config.path_prefix}/{pack_id}.bin"
    store = settings_crud.object_store(session)
    store.upload(path, data)
    create_bucket_files(
        [
            BucketFileRecord(
                id=pack_id,
                kind=BucketKind.WAVEFORM,
                path=path,
                size=len(data),
            )
        ]
    )
    return publish_waveform(
        AudioWaveformRecord(
            audio_file_id=audio_file_id,
            updated_at=datetime.now(UTC),
            pack_id=pack_id,
            byte_offset=0,
            byte_length=len(data),
            duration=duration,
            sample_rate=payload.sample_rate,
            points_per_second=payload.points_per_second,
            point_count=len(payload.peaks),
        )
    )


def replace_waveform_from_audio(
    session: Session,
    audio_file_id: uuid.UUID,
    audio_bytes: bytes,
    duration: float,
    payload: WaveformInput | None,
) -> AudioWaveformRecord:
    waveform = payload if payload is not None else _waveform_from_audio(audio_bytes)
    return replace_waveform(session, audio_file_id, duration, waveform)


def read_waveform(
    session: Session,
    audio_file_id: uuid.UUID,
    start: float,
    end: float,
    max_points: int,
) -> WaveformRead:
    item = get_waveform(audio_file_id)
    pack = get_bucket_file(item.pack_id)
    first = max(0, min(item.point_count, int(start * item.points_per_second)))
    last = max(
        first + 1,
        min(item.point_count, int(end * item.points_per_second) + 1),
    )
    data = settings_crud.object_store(session).read_range(
        ObjectRange(
            pack.path,
            item.byte_offset + first * 4,
            (last - first) * 4,
        )
    )
    return WaveformRead(
        duration=item.duration,
        sample_rate=item.sample_rate,
        points_per_second=item.points_per_second,
        start=first / item.points_per_second,
        end=last / item.points_per_second,
        peaks=downsample(decode_peaks(data), max_points),
    )


def delete_waveform(
    session: Session,
    audio_file_id: uuid.UUID,
    commit: bool = True,
) -> None:
    bulk_delete_waveforms(session, [audio_file_id], commit)


def bulk_delete_waveforms(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
    commit: bool = True,
) -> None:
    del commit
    ids = list(dict.fromkeys(audio_file_ids))
    waveforms = get_waveforms(ids)
    delete_waveforms(ids)
    delete_unreferenced_bucket_files(session, [item.pack_id for item in waveforms])


def waveform_exists(audio_file_id: uuid.UUID) -> bool:
    try:
        get_waveform(audio_file_id)
    except KeyError:
        return False
    return True


def _waveform_from_audio(data: bytes) -> WaveformInput:
    try:
        return waveform_from_wav(data)
    except (EOFError, ValueError, wave.Error) as error:
        raise ValueError("Waveform is required for non-WAV audio bytes") from error
