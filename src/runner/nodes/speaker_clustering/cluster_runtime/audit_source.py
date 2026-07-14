from __future__ import annotations

from typing import Any
from uuid import UUID

from runflow.core.node import Node
from runflow.core.settings import StrictSettings
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import SpeakerAuditRefPort
from runner.nodes.models import SpeakerAuditRef
from shared.db import database_session
from shared.db.reviews import crud as review_crud
from shared.db.reviews.schemas import ReviewState
from shared.db.speakers import crud as speaker_crud
from shared.db.speakers.schemas import SpeakerAuditState


class SpeakerAuditSourceSettings(StrictSettings):
    audit_id: UUID


class SpeakerAuditSourceNode(Node):
    NODE_TYPE = "SpeakerAuditSource"
    DESCRIPTION = "Load one approved, completed speaker audit for assignment."
    CATEGORY = "Speaker Clustering"
    SETTINGS = SpeakerAuditSourceSettings
    INPUTS = {}
    OUTPUTS = {"audit": SpeakerAuditRefPort()}
    IS_INPUT = True
    BATCH_POLICY = BatchPolicy(BatchMode.DISABLED)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=False)

    def __init__(self, node_id: str | None = None, **params: Any) -> None:
        super().__init__(node_id=node_id, **params)
        self._emitted = False

    def remaining_items(self, context: Any) -> int:
        return 0 if self._emitted else 1

    async def execute(
        self, batch: list[dict[str, Any]], context: Any
    ) -> list[dict[str, Any]]:
        if self._emitted:
            return []
        with database_session() as session:
            audit = speaker_crud.get_audit(session, self.settings.audit_id)
            if audit.state != SpeakerAuditState.COMPLETED.value:
                raise ValueError(f"speaker audit {audit.id} is {audit.state}")
            assert audit.review_id is not None
            review = review_crud.get_review(session, audit.review_id)
            if review.state != ReviewState.APPROVED.value:
                raise ValueError(f"workflow review {review.id} is {review.state}")
            result = SpeakerAuditRef(audit.id, audit.cluster_run_id, review.id)
        self._emitted = True
        return [{"audit": result}]
