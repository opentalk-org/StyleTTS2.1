from __future__ import annotations

from typing import Any

from fastapi import WebSocket


class WebSocketHub:
    def __init__(self) -> None:
        self._global_clients: set[WebSocket] = set()

    async def connect_global(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._global_clients.add(websocket)

    def disconnect_global(self, websocket: WebSocket) -> None:
        self._global_clients.discard(websocket)

    async def broadcast_global(self, payload: dict[str, Any]) -> None:
        clients = list(self._global_clients)
        for websocket in clients:
            await websocket.send_json(payload)
