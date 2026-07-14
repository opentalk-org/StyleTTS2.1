from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from backend.reviews.schemas import ReviewDecisionRequest, ReviewDecisionResponse
from backend.reviews.service import decide_review
from backend.service import BackendManager
from shared.db import database_session
from shared.db.reviews import crud as review_crud
from shared.db.reviews.schemas import ReviewRead, ReviewSummary


def review_router(manager: BackendManager) -> APIRouter:
    router = APIRouter(prefix="/reviews", tags=["reviews"])

    @router.get("", response_model=list[ReviewSummary])
    async def list_reviews(
        run_id: str = Query(min_length=1),
    ) -> list[ReviewSummary]:
        with database_session() as session:
            return [
                ReviewSummary.model_validate(item)
                for item in review_crud.list_reviews_for_run(session, run_id)
            ]

    @router.get("/{review_id}", response_model=ReviewRead)
    async def get_review(review_id: UUID) -> ReviewRead:
        try:
            with database_session() as session:
                return ReviewRead.model_validate(
                    review_crud.get_review(session, review_id)
                )
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error

    @router.post(
        "/{review_id}/decision",
        response_model=ReviewDecisionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def post_decision(
        review_id: UUID, payload: ReviewDecisionRequest
    ) -> ReviewDecisionResponse:
        try:
            return await decide_review(manager, review_id, payload.decision)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(error)
            ) from error

    return router
