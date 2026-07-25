from pydantic import BaseModel


class InitializationCreate(BaseModel):
    is_initialized: bool
