from pathlib import Path

from shared.db.jobs.models import Job
from shared.schemas import RunState, RunStatus


def job_status(job: Job) -> RunStatus:
    event_count = int(job.snapshot["total_event_count"]) if job.snapshot is not None else 0
    return RunStatus(
        run_id=job.run_id,
        state=RunState(job.state),
        workflow_path=Path("inline_graph"),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        event_count=event_count,
    )
