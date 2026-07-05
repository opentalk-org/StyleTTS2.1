from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.staticfiles import StaticFiles

from backend.nats_bus import BackendNatsBus
from backend.service import BackendManager, DuplicateRunError
from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runflow.tmp_nodes.audio.datatypes import register_audio_types
from runflow.tmp_nodes.register import register_builtin_nodes
from runflow.ui.schema_export import export_ui_schema
from shared.schemas import InlineGraphRunRequest, RunEventResponse, RunnerStatus, RunSnapshot, RunStatus


manager = BackendManager()
nats_bus = BackendNatsBus(manager)
manager.set_command_bus(nats_bus)
static_dir = Path(__file__).parent / "ui" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await nats_bus.start()
    try:
        yield
    finally:
        await nats_bus.stop()


app = FastAPI(title="Runflow Backend", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
async def schema() -> dict:
    node_registry = register_builtin_nodes(NodeRegistry())
    type_registry = register_audio_types(TypeRegistry())
    return export_ui_schema(node_registry, type_registry)


@app.post("/graphs/runs", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_graph_run(request: InlineGraphRunRequest) -> RunStatus:
    try:
        return await manager.start_inline_graph(request)
    except DuplicateRunError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.get("/runs", response_model=RunnerStatus)
async def list_runs() -> RunnerStatus:
    return await manager.list_statuses()


@app.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    try:
        return await manager.status(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
async def get_run_snapshot(run_id: str) -> RunSnapshot:
    try:
        return await manager.snapshot(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.get("/runs/{run_id}/events", response_model=list[RunEventResponse])
async def get_run_events(run_id: str, after: int = 0) -> list[RunEventResponse]:
    try:
        return await manager.events(run_id, after)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.get("/runs/{run_id}/errors", response_model=list[RunEventResponse])
async def get_run_errors(run_id: str) -> list[RunEventResponse]:
    try:
        return await manager.errors(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.post("/runs/{run_id}/stop", response_model=RunStatus)
async def stop_run(run_id: str) -> RunStatus:
    try:
        return await manager.stop(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.websocket("/ws")
async def backend_socket(websocket: WebSocket) -> None:
    try:
        await manager.connect_socket(websocket)
    except WebSocketDisconnect:
        return
