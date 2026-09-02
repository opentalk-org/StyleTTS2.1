from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AudioWaveformRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio_file_id: UUID
    updated_at: datetime
    pack_id: UUID
    byte_offset: int
    byte_length: int
    duration: float
    sample_rate: int
    points_per_second: int
    point_count: int
