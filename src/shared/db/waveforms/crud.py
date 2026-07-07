import uuid
import wave
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from shared.db.settings import crud as settings_crud
from shared.db.waveforms.codec import FORMAT_VERSION, decode_peaks, downsample, encode_peaks, waveform_from_wav
from shared.db.waveforms.models import AudioWaveform
from shared.db.waveforms.pack_store import ObjectStore, WaveformPackConfig, WaveformPackWriter
from shared.db.waveforms.schemas import WaveformInput, WaveformRead
from shared.storage import S3ObjectStore


def replace_waveform(
    session: Session,
    audio_file_id: uuid.UUID,
    duration: float,
    payload: WaveformInput,
    store: ObjectStore | None = None,
    config: WaveformPackConfig = WaveformPackConfig(),
) -> AudioWaveform:
    delete_waveform(session, audio_file_id, commit=False)
    data = encode_peaks(payload.peaks)
    writer = WaveformPackWriter(session, _object_store(session, store), config)
    write = writer.append(data)
    item = AudioWaveform(
        audio_file_id=audio_file_id,
        pack_id=write.pack.id,
        byte_offset=write.byte_offset,
        byte_length=write.byte_length,
        duration=duration,
        sample_rate=payload.sample_rate,
        points_per_second=payload.points_per_second,
        point_count=len(payload.peaks),
        format_version=FORMAT_VERSION,
        updated_at=_now(),
    )
    writer.flush()
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


def replace_waveform_from_audio(
    session: Session,
    audio_file_id: uuid.UUID,
    audio_bytes: bytes,
    duration: float,
    payload: WaveformInput | None,
    store: ObjectStore | None = None,
) -> AudioWaveform:
    waveform = payload if payload is not None else _waveform_from_audio(audio_bytes)
    return replace_waveform(session, audio_file_id, duration, waveform, store)


def read_waveform(
    session: Session,
    audio_file_id: uuid.UUID,
    start: float,
    end: float,
    max_points: int,
    store: ObjectStore | None = None,
) -> WaveformRead:
    item = session.get(AudioWaveform, audio_file_id)
    if item is None:
        raise KeyError(f"Waveform not found: {audio_file_id}")
    first = max(0, min(item.point_count, int(start * item.points_per_second)))
    last = max(first + 1, min(item.point_count, int(end * item.points_per_second) + 1))
    offset = item.byte_offset + first * 4
    data = _object_store(session, store).read_range(item.pack.path, offset, (last - first) * 4)
    peaks = downsample(decode_peaks(data), max_points)
    return WaveformRead(
        duration=item.duration,
        sample_rate=item.sample_rate,
        points_per_second=item.points_per_second,
        start=first / item.points_per_second,
        end=last / item.points_per_second,
        peaks=peaks,
    )


def delete_waveform(session: Session, audio_file_id: uuid.UUID, commit: bool = True) -> None:
    item = session.get(AudioWaveform, audio_file_id)
    if item is None:
        return
    item.pack.used_bytes -= item.byte_length
    assert item.pack.used_bytes >= 0, f"waveform pack used bytes went negative: {item.pack_id}"
    session.delete(item)
    if commit:
        session.commit()


def _waveform_from_audio(data: bytes) -> WaveformInput:
    try:
        return waveform_from_wav(data)
    except (EOFError, ValueError, wave.Error) as error:
        raise ValueError("Waveform is required for non-WAV audio bytes") from error


def _object_store(session: Session, store: ObjectStore | None) -> ObjectStore:
    if store is not None:
        return store
    return S3ObjectStore(settings_crud.object_store_config(session))


def _now() -> datetime:
    return datetime.now(UTC)
