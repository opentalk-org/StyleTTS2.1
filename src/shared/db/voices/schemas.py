from uuid import UUID

from pydantic import BaseModel, ConfigDict


class VoiceCreate(BaseModel):
    name: str


class VoiceRead(VoiceCreate):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
