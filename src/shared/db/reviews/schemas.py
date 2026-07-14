from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.schemas import InlineGraphRunRequest


class ReviewState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


ReviewTone = Literal["neutral", "success", "warning", "danger"]


class ReviewMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str
    numeric_value: float | None
    tone: ReviewTone


class ReviewField(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str


class AudioSegmentReviewMedia(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["audio_segment"]
    audio_file_id: UUID
    segment_id: str = Field(min_length=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "AudioSegmentReviewMedia":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("audio review media end must be after start")
        return self


ReviewMedia = Annotated[AudioSegmentReviewMedia, Field(discriminator="kind")]


class ReviewItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    subtitle: str | None
    fields: tuple[ReviewField, ...] = Field(max_length=32)
    media: tuple[ReviewMedia, ...] = Field(max_length=8)


class ReviewGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    explanation: str
    tone: ReviewTone
    items: tuple[ReviewItem, ...] = Field(max_length=1000)


class ReviewPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: tuple[ReviewMetric, ...] = Field(max_length=64)
    warnings: tuple[str, ...] = Field(max_length=64)
    groups: tuple[ReviewGroup, ...] = Field(max_length=32)


class ReviewContinuation(BaseModel):
    model_config = ConfigDict(frozen=True)

    graph: InlineGraphRunRequest

    @model_validator(mode="after")
    def validate_unassigned_run(self) -> "ReviewContinuation":
        if self.graph.run_id is not None:
            raise ValueError("review continuation run_id must be unassigned")
        return self


class ReviewCreate(BaseModel):
    producer_run_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    payload: ReviewPayload
    continuation: ReviewContinuation | None


class ReviewRead(ReviewCreate):
    id: UUID
    state: ReviewState
    continuation_run_id: str | None
    created_at: datetime
    decided_at: datetime | None
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ReviewSummary(BaseModel):
    id: UUID
    producer_run_id: str
    kind: str
    title: str
    state: ReviewState
    continuation_run_id: str | None
    created_at: datetime
    decided_at: datetime | None
    model_config = ConfigDict(from_attributes=True)
