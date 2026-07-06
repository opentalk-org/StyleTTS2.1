from typing import Any

from fastapi import WebSocket, WebSocketDisconnect


class WebSocketHub:
    def __init__(self) -> None:
        self._global_clients: set[WebSocket] = set()
        self._client_run: dict[WebSocket, str] = {}
        self._run_clients: dict[str, set[WebSocket]] = {}

    async def connect_global(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._global_clients.add(websocket)

    def disconnect_global(self, websocket: WebSocket) -> None:
        self._global_clients.discard(websocket)
        self.watch_run(websocket, None)

    async def broadcast_global(self, payload: dict[str, Any]) -> None:
        await self._send(list(self._global_clients), payload)

    async def broadcast_run(self, run_id: str, payload: dict[str, Any]) -> None:
        await self._send(list(self._run_clients.get(run_id, set())), payload)

    def watch_run(self, websocket: WebSocket, run_id: str | None) -> None:
        previous = self._client_run.pop(websocket, None)
        if previous is not None and previous in self._run_clients:
            self._run_clients[previous].discard(websocket)
        if run_id is None:
            return
        self._client_run[websocket] = run_id
        if run_id not in self._run_clients:
            self._run_clients[run_id] = set()
        self._run_clients[run_id].add(websocket)

    async def _send(self, clients: list[WebSocket], payload: dict[str, Any]) -> None:
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except WebSocketDisconnect:
                self.disconnect_global(websocket)
