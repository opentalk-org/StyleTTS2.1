from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import bindparam, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from shared.db.jobs.models import Job, NodeLog, RunNodeState
from shared.db.jobs.schemas import ClaimedJob, RunnerStateFlush
from shared.db.runners.models import Runner
from shared.run_snapshots import failed_run_snapshot, stopped_run_snapshot
from shared.schemas import RunSnapshot


LEASE_SECONDS = 120

RUN_MAX_ATTEMPTS = 3
RUN_NOTIFICATION_CHANNEL = "runflow_runs"
RUNNER_NOTIFICATION_CHANNEL = "runflow_runners"


def claim_jobs(session: Session, runner_id: str, limit: int) -> list[ClaimedJob]:
    now = datetime.now(UTC)
    recovered = _recover_expired_claims(session, now)
    candidates = (
        select(Job.run_id)
        .where(
            Job.state == "queued",
            Job.desired_state == "running",
            or_(Job.target_runner_id.is_(None), Job.target_runner_id == runner_id),
        )
        .order_by(Job.created_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
        .cte("claimable_jobs")
    )
    rows = session.execute(
        Job.__table__.update()
        .where(Job.run_id.in_(select(candidates.c.run_id)))
        .values(
            state="running",
            claimed_runner_id=runner_id,
            started_at=func.coalesce(Job.started_at, now),
            lease_expires_at=now + timedelta(seconds=LEASE_SECONDS),
            updated_at=now,
        )
        .returning(Job.run_id, Job.graph_request)
    ).all()
    if recovered or rows:
        _notify(session, RUN_NOTIFICATION_CHANNEL)
    session.commit()
    return [ClaimedJob(run_id=row.run_id, graph_request=row.graph_request) for row in rows]


def set_desired_job_state(session: Session, run_id: str, desired_state: str) -> Job:
    item = session.get(Job, run_id)
    if item is None:
        raise KeyError(f"Job not found: {run_id}")
    item.desired_state = desired_state
    if desired_state == "stopped" and item.state == "queued":
        item.state = "stopped"
        item.finished_at = datetime.now(UTC)
    elif desired_state == "stopped" and item.state == "running":
        item.state = "stopping"
    item.updated_at = datetime.now(UTC)
    _notify(session, RUN_NOTIFICATION_CHANNEL)
    session.commit()
    session.refresh(item)
    return item


def desired_job_states(session: Session, runner_id: str, run_ids: Sequence[str]) -> dict[str, str]:
    rows = session.execute(
        select(Job.run_id, Job.desired_state).where(
            Job.claimed_runner_id == runner_id,
            Job.run_id.in_(run_ids),
        )
    ).all()
    return {row.run_id: row.desired_state for row in rows}


def set_desired_node_state(session: Session, run_id: str, node_id: str, desired_loaded: bool) -> None:
    now = datetime.now(UTC)
    statement = insert(RunNodeState).values(
        run_id=run_id,
        node_id=node_id,
        desired_loaded=desired_loaded,
        observed_loaded=None,
        error=None,
        updated_at=now,
    )
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_run_node_states_run_node",
            set_={"desired_loaded": desired_loaded, "error": None, "updated_at": now},
        )
    )
    _notify(session, RUN_NOTIFICATION_CHANNEL)
    session.commit()


def pending_node_states(session: Session, run_ids: Sequence[str]) -> list[RunNodeState]:
    return list(
        session.execute(
            select(RunNodeState).where(
                RunNodeState.run_id.in_(run_ids),
                or_(
                    RunNodeState.observed_loaded.is_(None),
                    RunNodeState.observed_loaded != RunNodeState.desired_loaded,
                ),
            )
        ).scalars()
    )


def flush_runner_state(session: Session, payload: RunnerStateFlush) -> None:
    now = datetime.now(UTC)
    _upsert_runner(session, payload, now)
    _update_jobs(session, payload, now)
    _upsert_node_states(session, payload, now)
    _upsert_logs(session, payload, now)
    _notify(session, RUNNER_NOTIFICATION_CHANNEL)
    if payload.jobs or payload.node_states or payload.logs:
        _notify(session, RUN_NOTIFICATION_CHANNEL)
    session.commit()


def _upsert_runner(session: Session, payload: RunnerStateFlush, now: datetime) -> None:
    item = session.execute(select(Runner).where(Runner.name == payload.runner_id)).scalar_one_or_none()
    values = {
        "hostname": payload.hostname,
        "port": payload.port,
        "gpu_index": payload.gpu_index,
        "process_id": payload.process_id,
        "active_run_ids": payload.active_run_ids,
        "capabilities": payload.capabilities,
        "last_seen_at": now,
    }
    if item is None:
        session.add(Runner(name=payload.runner_id, **values))
        return
    for key, value in values.items():
        setattr(item, key, value)


def _update_jobs(session: Session, payload: RunnerStateFlush, now: datetime) -> None:
    if payload.active_run_ids:
        session.execute(
            update(Job)
            .where(Job.claimed_runner_id == payload.runner_id, Job.run_id.in_(payload.active_run_ids))
            .values(lease_expires_at=now + timedelta(seconds=LEASE_SECONDS))
        )
    if not payload.jobs:
        return
    statement = (
        Job.__table__.update()
        .where(Job.run_id == bindparam("job_run_id"), Job.claimed_runner_id == payload.runner_id)
        .values(
            state=bindparam("job_state"),
            snapshot=bindparam("job_snapshot"),
            started_at=bindparam("job_started_at"),
            finished_at=bindparam("job_finished_at"),
            error=bindparam("job_error"),
            claimed_runner_id=bindparam("job_claimed_runner_id"),
            lease_expires_at=bindparam("job_lease_expires_at"),
            updated_at=now,
        )
    )
    session.execute(
        statement,
        [
            {
                "job_run_id": job.run_id,
                "job_state": job.state,
                "job_snapshot": job.snapshot,
                "job_started_at": job.started_at,
                "job_finished_at": job.finished_at,
                "job_error": job.error,
                "job_claimed_runner_id": None if job.release_claim else payload.runner_id,
                "job_lease_expires_at": None if job.release_claim else now + timedelta(seconds=LEASE_SECONDS),
            }
            for job in payload.jobs
        ],
    )


def _upsert_node_states(session: Session, payload: RunnerStateFlush, now: datetime) -> None:
    if not payload.node_states:
        return
    values = [state.model_dump() | {"updated_at": now} for state in payload.node_states]
    statement = insert(RunNodeState).values(values)
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_run_node_states_run_node",
            set_={
                "desired_loaded": statement.excluded.desired_loaded,
                "observed_loaded": statement.excluded.observed_loaded,
                "error": statement.excluded.error,
                "updated_at": now,
            },
        )
    )


def _upsert_logs(session: Session, payload: RunnerStateFlush, now: datetime) -> None:
    if not payload.logs:
        return
    values = [item.model_dump() | {"updated_at": now} for item in payload.logs]
    statement = insert(NodeLog).values(values)
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_node_logs_run_node",
            set_={
                "content": statement.excluded.content,
                "truncated": statement.excluded.truncated,
                "error": statement.excluded.error,
                "updated_at": now,
            },
        )
    )


def _notify(session: Session, channel: str) -> None:
    session.execute(select(func.pg_notify(channel, "")))


def _recover_expired_claims(session: Session, now: datetime) -> bool:
    expired = (
        Job.state.in_({"running", "stopping"}),
        Job.lease_expires_at < now,
    )
    running_expired = list(
        session.execute(
            select(Job)
            .where(*expired, Job.desired_state == "running")
            .with_for_update(skip_locked=True)
        ).scalars()
    )
    requeued = _recover_running_jobs(running_expired, now)
    stopping_jobs = list(
        session.execute(
            select(Job)
            .where(*expired, Job.desired_state == "stopped")
            .with_for_update(skip_locked=True)
        ).scalars()
    )
    for job in stopping_jobs:
        if job.snapshot is not None:
            snapshot = RunSnapshot.model_validate(job.snapshot)
            job.snapshot = stopped_run_snapshot(
                snapshot,
                "run stopped after the runner lease expired",
            ).model_dump(mode="json")
        job.state = "stopped"
        job.claimed_runner_id = None
        job.lease_expires_at = None
        job.finished_at = now
        job.updated_at = now
    return requeued > 0 or bool(stopping_jobs)


def _recover_running_jobs(jobs: list[Job], now: datetime) -> int:
    """Re-queue crashed runs, but fail those that have exhausted their attempts."""
    for job in jobs:
        job.attempts += 1
        job.updated_at = now
        if job.attempts >= RUN_MAX_ATTEMPTS:
            message = (
                f"Runner terminated unexpectedly {job.attempts} times "
                "(out of memory or crash); giving up"
            )
            if job.snapshot is not None:
                snapshot = RunSnapshot.model_validate(job.snapshot)
                job.snapshot = failed_run_snapshot(snapshot, message).model_dump(mode="json")
            job.state = "failed"
            job.error = message
            job.finished_at = now
            job.claimed_runner_id = None
            job.lease_expires_at = None
        else:
            job.state = "queued"
            job.claimed_runner_id = None
            job.lease_expires_at = None
    return len(jobs)
