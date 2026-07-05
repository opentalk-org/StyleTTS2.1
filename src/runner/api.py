from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles

from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runflow.tmp_nodes.audio.datatypes import register_audio_types
from runflow.tmp_nodes.register import register_builtin_nodes
from runflow.ui.schema_export import export_ui_schema
from runner.graphs import InlineGraphRunRequest
from runner.schemas import RunnerStatus, RunEventResponse, RunStartRequest, RunState, RunStatus
from runner.service import DuplicateRunError, RunnerManager


app = FastAPI(title="Runflow Runner")
manager = RunnerManager()
static_dir = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
async def schema() -> dict:
    node_registry = register_builtin_nodes(NodeRegistry())
    type_registry = register_audio_types(TypeRegistry())
    return export_ui_schema(node_registry, type_registry)


@app.post("/runs", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_run(request: RunStartRequest) -> RunStatus:
    try:
        return await manager.start(request)
    except DuplicateRunError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.post("/graphs/runs", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_graph_run(request: InlineGraphRunRequest) -> RunStatus:
    try:
        return await manager.start_inline_graph(request)
    except DuplicateRunError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.get("/runs", response_model=RunnerStatus)
async def list_runs() -> RunnerStatus:
    runs = await manager.list_statuses()
    active_states = {RunState.QUEUED, RunState.RUNNING, RunState.STOPPING}
    active_runs = [run for run in runs if run.state in active_states]
    return RunnerStatus(total_runs=len(runs), active_runs=len(active_runs), runs=runs)


@app.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(run_id: str) -> RunStatus:
    try:
        return await manager.status(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.get("/runs/{run_id}/events", response_model=list[RunEventResponse])
async def get_run_events(run_id: str, after: int = 0) -> list[RunEventResponse]:
    try:
        return await manager.events(run_id, after)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.post("/runs/{run_id}/stop", response_model=RunStatus)
async def stop_run(run_id: str) -> RunStatus:
    try:
        return await manager.stop(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
