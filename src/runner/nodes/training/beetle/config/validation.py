from pydantic import Field

from .architecture import StrictConfigModel


class ValidationConfig(StrictConfigModel):
    sample_count: int = Field(gt=0)
