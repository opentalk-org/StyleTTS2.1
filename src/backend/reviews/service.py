from __future__ import annotations

from uuid import UUID

from backend.reviews.schemas import ReviewDecisionResponse
from backend.service import BackendManager, DuplicateRunError
from shared.db import database_session
from shared.db.reviews import crud as review_crud
from shared.db.reviews.schemas import ReviewDecision, ReviewRead


async def decide_review(
    manager: BackendManager,
    review_id: UUID,
    decision: ReviewDecision,
) -> ReviewDecisionResponse:
    with database_session() as session:
        review = review_crud.decide_review(session, review_id, decision)
        continuation = review_crud.review_continuation(review)
        response_review = ReviewRead.model_validate(review)
    if decision == ReviewDecision.REJECTED or continuation is None:
        return ReviewDecisionResponse(
            review=response_review,
            continuation_run_id=None,
            continuation=None,
        )
    assert review.continuation_run_id is not None
    request = continuation.graph.model_copy(
        update={"run_id": review.continuation_run_id}
    )
    try:
        status = await manager.start_inline_graph(
            request, name=f"Review: {review.title}"
        )
    except DuplicateRunError:
        status = await manager.status(review.continuation_run_id)
    return ReviewDecisionResponse(
        review=response_review,
        continuation_run_id=review.continuation_run_id,
        continuation=status,
    )
