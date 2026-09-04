from uuid import UUID

from pydantic import BaseModel, ConfigDict


class StorageSettingsPayload(BaseModel):
    bucket: str = "runflow"
    folder: str = "/"
    endpoint_url: str = "http://127.0.0.1:9001"
    region_name: str = "us-east-1"
    access_key_id: str = "runflow"
    secret_access_key: str = "runflow-secret"


class StorageSettingsRead(StorageSettingsPayload):
    id: UUID
    model_config = ConfigDict(from_attributes=True)


class IntegrationSettingsPayload(BaseModel):
    hf_token: str = ""
    openrouter_token: str = ""
    mlflow_url: str = "http://localhost:7860"


class IntegrationSettingsRead(IntegrationSettingsPayload):
    id: UUID
    model_config = ConfigDict(from_attributes=True)
