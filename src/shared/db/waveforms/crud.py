import uuid
import wave
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from shared.db.settings import crud as settings_crud
from shared.db.waveforms.codec import FORMAT_VERSION, decode_peaks, downsample, encode_peaks, waveform_from_wav
from shared.db.waveforms.models import AudioWaveform, WaveformPack
from shared.db.waveforms.pack_store import ObjectStore, WaveformPackConfig, WaveformPackWriter
from shared.db.waveforms.schemas import WaveformInput, WaveformRead
from shared.storage import S3ObjectStore


def bulk_replace_waveforms_from_audio(
    session: Session,
    items: Sequence[tuple[uuid.UUID, bytes, float, WaveformInput | None]],
    store: ObjectStore | None = None,
    config: WaveformPackConfig = WaveformPackConfig(),
) -> list[AudioWaveform]:
    if not items:
        return []
    resolved_store = _object_store(session, store)
    writer = WaveformPackWriter(session, resolved_store, config)
    waveforms = []
    bulk_delete_waveforms(
        session,
        [audio_file_id for audio_file_id, _, _, _ in items],
        commit=False,
    )
    for audio_file_id, audio_bytes, duration, payload in items:
        waveform = payload if payload is not None else _waveform_from_audio(audio_bytes)
        data = encode_peaks(waveform.peaks)
        write = writer.append(data)
        waveforms.append(
            AudioWaveform(
                audio_file_id=audio_file_id,
                pack_id=write.pack.id,
                byte_offset=write.byte_offset,
                byte_length=write.byte_length,
                duration=duration,
                sample_rate=waveform.sample_rate,
                points_per_second=waveform.points_per_second,
                point_count=len(waveform.peaks),
                format_version=FORMAT_VERSION,
                updated_at=_now(),
            )
        )
    writer.flush()
    session.add_all(waveforms)
    session.commit()
    return waveforms


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
    bulk_delete_waveforms(session, [audio_file_id], commit=commit)


def bulk_delete_waveforms(
    session: Session,
    audio_file_ids: Sequence[uuid.UUID],
    commit: bool = True,
) -> None:
    ids = list(dict.fromkeys(audio_file_ids))
    if not ids:
        return
    rows = session.execute(
        select(AudioWaveform.pack_id, AudioWaveform.byte_length).where(
            AudioWaveform.audio_file_id.in_(ids)
        )
    ).all()
    removed_by_pack: dict[uuid.UUID, int] = {}
    for pack_id, byte_length in rows:
        removed_by_pack[pack_id] = removed_by_pack.get(pack_id, 0) + byte_length
    for pack_id, removed_bytes in removed_by_pack.items():
        result = session.execute(
            update(WaveformPack)
            .where(WaveformPack.id == pack_id, WaveformPack.used_bytes >= removed_bytes)
            .values(used_bytes=WaveformPack.used_bytes - removed_bytes)
        )
        assert result.rowcount == 1, f"waveform pack used bytes would go negative: {pack_id}"
    session.execute(delete(AudioWaveform).where(AudioWaveform.audio_file_id.in_(ids)))
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
