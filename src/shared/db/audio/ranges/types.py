from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class SegmentReadRequest:
    audio_file_id: UUID
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("segment read range must be positive and ordered")
