from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from shared.db.clickhouse.types import utc_datetime


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

    _updated_at_utc = field_validator("updated_at")(utc_datetime)
