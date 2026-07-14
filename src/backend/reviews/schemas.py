from pydantic import BaseModel

from shared.db.reviews.schemas import ReviewDecision, ReviewRead
from shared.schemas import RunStatus


class ReviewDecisionRequest(BaseModel):
    decision: ReviewDecision


class ReviewDecisionResponse(BaseModel):
    review: ReviewRead
    continuation_run_id: str | None
    continuation: RunStatus | None
