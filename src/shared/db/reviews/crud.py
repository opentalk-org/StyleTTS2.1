from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from shared.db.reviews.models import WorkflowReview
from shared.db.reviews.schemas import (
    ReviewContinuation,
    ReviewCreate,
    ReviewDecision,
    ReviewPayload,
    ReviewState,
)


def create_review(
    session: Session, payload: ReviewCreate, *, commit: bool = True
) -> WorkflowReview:
    existing = _review_for_identity(session, payload.kind, payload.source_key)
    if existing is not None:
        _validate_duplicate(existing, payload)
        return existing
    item = WorkflowReview(
        producer_run_id=payload.producer_run_id,
        kind=payload.kind,
        source_key=payload.source_key,
        title=payload.title,
        state=ReviewState.PENDING.value,
        payload=payload.payload.model_dump(mode="json"),
        continuation=(
            None
            if payload.continuation is None
            else payload.continuation.model_dump(mode="json")
        ),
    )
    session.add(item)
    try:
        if commit:
            session.commit()
        else:
            session.flush()
    except IntegrityError:
        session.rollback()
        existing = _review_for_identity(session, payload.kind, payload.source_key)
        if existing is None:
            raise
        _validate_duplicate(existing, payload)
        return existing
    if commit:
        session.refresh(item)
    return item


def list_reviews_for_run(session: Session, run_id: str) -> list[WorkflowReview]:
    return list(
        session.scalars(
            select(WorkflowReview)
            .where(WorkflowReview.producer_run_id == run_id)
            .order_by(WorkflowReview.created_at, WorkflowReview.id)
        )
    )


def get_review(session: Session, review_id: UUID) -> WorkflowReview:
    item = session.get(WorkflowReview, review_id)
    if item is None:
        raise KeyError(f"workflow review not found: {review_id}")
    return item


def decide_review(
    session: Session, review_id: UUID, decision: ReviewDecision
) -> WorkflowReview:
    item = session.scalar(
        select(WorkflowReview).where(WorkflowReview.id == review_id).with_for_update()
    )
    if item is None:
        raise KeyError(f"workflow review not found: {review_id}")
    if item.state != ReviewState.PENDING.value:
        if item.state == decision.value:
            return item
        raise ValueError(f"workflow review {review_id} is already {item.state}")
    now = datetime.now(UTC)
    item.state = decision.value
    item.decided_at = now
    item.updated_at = now
    if decision == ReviewDecision.APPROVED and item.continuation is not None:
        item.continuation_run_id = f"review_{item.id.hex}"
    session.commit()
    session.refresh(item)
    return item


def review_payload(item: WorkflowReview) -> ReviewPayload:
    return ReviewPayload.model_validate(item.payload)


def review_continuation(item: WorkflowReview) -> ReviewContinuation | None:
    if item.continuation is None:
        return None
    return ReviewContinuation.model_validate(item.continuation)


def _review_for_identity(
    session: Session, kind: str, source_key: str
) -> WorkflowReview | None:
    return session.scalar(
        select(WorkflowReview).where(
            WorkflowReview.kind == kind,
            WorkflowReview.source_key == source_key,
        )
    )


def _validate_duplicate(item: WorkflowReview, payload: ReviewCreate) -> None:
    stored = (
        item.producer_run_id,
        item.title,
        review_payload(item),
        review_continuation(item),
    )
    incoming = (
        payload.producer_run_id,
        payload.title,
        payload.payload,
        payload.continuation,
    )
    if stored != incoming:
        raise ValueError(
            f"workflow review {payload.kind}/{payload.source_key} has different payload"
        )
