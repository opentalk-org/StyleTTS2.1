from dataclasses import dataclass
from time import perf_counter
from typing import Protocol


class LineageEventEmitter(Protocol):
    async def lineage_started(self, lineage_id: str, source_node_id: str) -> None: ...

    async def lineage_completed(self, lineage_id: str, elapsed_ms: float) -> None: ...

    async def lineage_abandoned(self, lineage_id: str, reason: str) -> None: ...


@dataclass
class LineageState:
    source_node_id: str
    started_at: float
    source_open: int = 1
    tasks: int = 0
    joins: int = 0


class LineageTracker:
    def __init__(self, events: LineageEventEmitter):
        self.events = events
        self._active: dict[str, LineageState] = {}

    def tracks(self, lineage_id: str) -> bool:
        return lineage_id in self._active

    async def open_source(self, lineage_id: str, source_node_id: str) -> None:
        if lineage_id in self._active:
            raise RuntimeError(f"source lineage already active: {lineage_id}")
        self._active[lineage_id] = LineageState(source_node_id=source_node_id, started_at=perf_counter())
        await self.events.lineage_started(lineage_id, source_node_id)

    async def close_source(self, lineage_id: str) -> None:
        state = self._state(lineage_id)
        if state.source_open != 1:
            raise RuntimeError(f"source lineage is not open: {lineage_id}")
        state.source_open = 0
        await self._complete_if_idle(lineage_id, state)

    async def add_task(self, lineage_id: str) -> None:
        self._state(lineage_id).tasks += 1

    async def finish_task(self, lineage_id: str) -> None:
        state = self._state(lineage_id)
        if state.tasks <= 0:
            raise RuntimeError(f"source lineage has no active task: {lineage_id}")
        state.tasks -= 1
        await self._complete_if_idle(lineage_id, state)

    async def open_join(self, lineage_id: str) -> None:
        self._state(lineage_id).joins += 1

    async def close_join(self, lineage_id: str) -> None:
        state = self._state(lineage_id)
        if state.joins <= 0:
            raise RuntimeError(f"source lineage has no open join: {lineage_id}")
        state.joins -= 1
        await self._complete_if_idle(lineage_id, state)

    async def abandon_all(self, reason: str) -> None:
        active = tuple(self._active)
        for lineage_id in active:
            await self.events.lineage_abandoned(lineage_id, reason)
            del self._active[lineage_id]

    def _state(self, lineage_id: str) -> LineageState:
        try:
            return self._active[lineage_id]
        except KeyError as error:
            raise RuntimeError(f"unknown source lineage: {lineage_id}") from error

    async def _complete_if_idle(self, lineage_id: str, state: LineageState) -> None:
        if state.source_open or state.tasks or state.joins:
            return
        elapsed_ms = (perf_counter() - state.started_at) * 1000
        del self._active[lineage_id]
        await self.events.lineage_completed(lineage_id, elapsed_ms)
