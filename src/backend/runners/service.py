from datetime import UTC, datetime

from shared.schemas import RunnerHeartbeatMessage


STALE_AFTER_SECONDS = 15


class RunnerLiveRegistry:
    def __init__(self) -> None:
        self._heartbeats: dict[str, RunnerHeartbeatMessage] = {}

    def record(self, heartbeat: RunnerHeartbeatMessage) -> None:
        self._heartbeats[heartbeat.runner_id] = heartbeat

    def heartbeat(self, runner_id: str) -> RunnerHeartbeatMessage | None:
        return self._heartbeats[runner_id] if runner_id in self._heartbeats else None

    def is_online(self, heartbeat: RunnerHeartbeatMessage | None) -> bool:
        if heartbeat is None:
            return False
        age = datetime.now(UTC) - heartbeat.created_at
        return age.total_seconds() <= STALE_AFTER_SECONDS


runner_live_registry = RunnerLiveRegistry()
