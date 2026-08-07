from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class TimestampSnapshot:
    audio_file_id: UUID
    audio_timestamps: Any
    parakeet_timestamps: Any


@dataclass(frozen=True)
class CleanupAudit:
    audio_file_id: UUID
    before: TimestampSnapshot
    after: TimestampSnapshot


@dataclass(frozen=True)
class CleanupBatchResult:
    last_audio_file_id: UUID
    examined: int
    pruned: int
    already_pruned: int
    missing_parakeet: int
    audits: tuple[CleanupAudit, ...]
