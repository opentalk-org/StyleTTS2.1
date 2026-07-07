from __future__ import annotations

from typing import Any

from runner.nodes.models import Audio, AudioSegment


DEFAULT_MODEL_PRIORITY = ("src", "canary", "parakeet", "whisper")


def speech_segment_records(audio: Audio) -> dict[str, Any]:
    groups: dict[tuple[int, int], list[AudioSegment]] = {}
    for segment in audio.segments:
        key = (round(segment.start * 1000.0), round(segment.end * 1000.0))
        groups.setdefault(key, []).append(segment)
    records: list[dict[str, Any]] = []
    collapsed = 0
    for key in sorted(groups):
        members = groups[key]
        collapsed += len(members) - 1
        records.append(_canonical_record(audio, members))
    return {
        "audio_file_id": str(audio.audio_file_id),
        "segments": records,
        "duplicate_segments_collapsed": collapsed,
    }


def _canonical_record(audio: Audio, members: list[AudioSegment]) -> dict[str, Any]:
    canonical = _select_canonical(members)
    phon = canonical.phon.strip()
    if not phon:
        phon = next((member.phon.strip() for member in members if member.phon.strip()), "")
    return {
        "source_audio_id": str(audio.audio_file_id),
        "text": canonical.text.strip(),
        "phon": phon,
        "speaker": (canonical.speaker or "").strip(),
        "start": float(canonical.start),
        "end": float(canonical.end),
        "duration": float(canonical.duration),
        "model": _segment_model(canonical),
    }


def _select_canonical(members: list[AudioSegment]) -> AudioSegment:
    preferred_column = _preferred_column(members)
    if preferred_column is not None:
        for member in members:
            if str(member.metadata.get("text_column", "")) == preferred_column:
                return member
        preferred_model = preferred_column.removeprefix("text_")
        for member in members:
            if _segment_model(member) == preferred_model:
                return member
    ranked = sorted(members, key=lambda member: (_model_rank(member), -len(member.text.strip())))
    return ranked[0]


def _preferred_column(members: list[AudioSegment]) -> str | None:
    for member in members:
        value = member.metadata.get("preferred_text_column")
        if value:
            return str(value)
    return None


def _model_rank(member: AudioSegment) -> int:
    model = _segment_model(member)
    if model in DEFAULT_MODEL_PRIORITY:
        return DEFAULT_MODEL_PRIORITY.index(model)
    return len(DEFAULT_MODEL_PRIORITY)


def _segment_model(member: AudioSegment) -> str:
    for field in ("model", "type_"):
        value = member.metadata.get(field)
        if value:
            return str(value)
    return ""
