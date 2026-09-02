import base64
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shared.db.audio.clickhouse.models import AudioFileRecord


@dataclass(frozen=True)
class AudioCursor:
    sort: str
    value: Any
    audio_file_id: uuid.UUID

    def encode(self) -> str:
        value = (
            self.value.isoformat() if isinstance(self.value, datetime) else self.value
        )
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


def cursor_for_row(sort: str, item: AudioFileRecord) -> AudioCursor:
    values = {"updated": item.updated_at, "duration": item.duration}
    return AudioCursor(sort=sort, value=values[sort], audio_file_id=item.id)
