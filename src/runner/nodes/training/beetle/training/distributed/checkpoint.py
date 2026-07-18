from dataclasses import replace
from pathlib import Path

from ..checkpoint import CheckpointManager, CheckpointPayload
from ..state import RankState
from .runtime import DistributedRuntime


class DistributedCheckpointManager:
    def __init__(
        self,
        local: CheckpointManager,
        runtime: DistributedRuntime,
    ) -> None:
        self.local = local
        self.runtime = runtime
        self.root = local.root

    def save(self, payload: CheckpointPayload) -> Path:
        if len(payload.rank_states) != 1:
            raise ValueError("process-local checkpoint must contain one rank state")
        gathered = self.runtime.gather_objects(payload.rank_states[0])
        rank_states = tuple(
            state if isinstance(state, RankState) else _raise_rank_state_type()
            for state in gathered
        )
        path = (
            self.local.save(replace(payload, rank_states=rank_states))
            if self.runtime.is_main_process
            else None
        )
        shared = self.runtime.broadcast_object(path)
        if not isinstance(shared, Path):
            raise TypeError("main process did not broadcast a checkpoint path")
        return shared


def _raise_rank_state_type() -> RankState:
    raise TypeError("distributed checkpoint gathered an invalid rank state")
