from uuid import UUID

from pydantic import BaseModel, ConfigDict


class VoiceCreate(BaseModel):
    name: str


class VoiceRead(VoiceCreate):
    id: UUID
    segments: int
    datasets: list[UUID]
    model_config = ConfigDict(from_attributes=True)


class VoicePage(BaseModel):
    rows: list[VoiceRead]
    total: int
