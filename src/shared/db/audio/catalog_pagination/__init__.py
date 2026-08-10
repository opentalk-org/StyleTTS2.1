import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import and_, or_

from shared.db.audio.models import AudioFile


@dataclass(frozen=True)
class AudioCursor:
    sort: str
    value: Any
    audio_file_id: uuid.UUID

    def encode(self) -> str:
        value = self.value.isoformat() if isinstance(self.value, datetime) else self.value
        payload = json.dumps(
            {"sort": self.sort, "value": value, "id": str(self.audio_file_id)},
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @classmethod
    def decode(cls, encoded: str, sort: str) -> "AudioCursor":
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
        if payload["sort"] != sort:
            raise ValueError("Audio cursor does not match the requested sort")
        value = payload["value"]
        if sort == "updated":
            value = datetime.fromisoformat(value)
        return cls(sort=sort, value=value, audio_file_id=uuid.UUID(payload["id"]))


def cursor_filter(cursor: AudioCursor):
    column, descending = _sort_column(cursor.sort)
    comparison = column < cursor.value if descending else column > cursor.value
    clauses = [comparison, and_(column == cursor.value, AudioFile.id > cursor.audio_file_id)]
    return or_(*clauses)


def cursor_for_row(sort: str, item: AudioFile, segment_count: int) -> AudioCursor:
    values = {
        "updated": item.updated_at,
        "name": item.name,
        "duration": item.duration,
        "segments": segment_count,
    }
    return AudioCursor(sort=sort, value=values[sort], audio_file_id=item.id)


def _sort_column(sort: str):
    if sort == "name":
        return AudioFile.name, False
    if sort == "duration":
        return AudioFile.duration, True
    if sort == "segments":
        return AudioFile.segment_count, True
    return AudioFile.updated_at, True
