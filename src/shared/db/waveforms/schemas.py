from pydantic import BaseModel


class WaveformInput(BaseModel):
    sample_rate: int
    points_per_second: int
    peaks: list[tuple[float, float]]


class WaveformRead(BaseModel):
    duration: float
    sample_rate: int
    points_per_second: int
    start: float
    end: float
    peaks: list[tuple[float, float]]
