from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncIterator
import uuid

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.audio import router as audio_router
from backend.runners import router as runners_router
from backend.settings import router as settings_router
from backend.nats_bus import BackendNatsBus
from backend.service import BackendManager, DuplicateRunError
from backend.workflows import workflow_router
from runflow.registry.node_registry import NodeRegistry
from runflow.registry.type_registry import TypeRegistry
from runflow.ui.schema_export import export_ui_schema
from runner.nodes.registry import register_runner_nodes, register_runner_types_for_ui
from shared.db import create_database_schema, database_session
from shared.db.datasets import crud as dataset_crud
from shared.db.datasets.models import Dataset
from shared.db.datasets.schemas import DatasetCreate, DatasetRead
from shared.db.voices import crud as voice_crud
from shared.db.voices.models import Voice
from shared.db.voices.schemas import VoiceCreate, VoicePage, VoiceRead
from shared.logging_setup import configure_logging, get_logger
from shared.schemas import InlineGraphRunRequest, NodeLogResponseMessage, RunEventResponse, RunnerStatus, RunSnapshot, RunStatus


configure_logging("backend")
logger = get_logger("backend.api")
manager = BackendManager()
nats_bus = BackendNatsBus(manager)
manager.set_command_bus(nats_bus)
old_static_dir = Path(__file__).parent / "ui" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("backend startup")
    create_database_schema()
    await nats_bus.start()
    try:
        yield
    finally:
        logger.info("backend shutdown")
        await nats_bus.stop()


app = FastAPI(title="Runflow Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui-old", StaticFiles(directory=old_static_dir, html=True), name="ui-old")
app.include_router(audio_router)
app.include_router(runners_router)
app.include_router(settings_router)
app.include_router(workflow_router(manager))

if "RUNFLOW_UI_STATIC_DIR" in os.environ:
    static_dir = Path(os.environ["RUNFLOW_UI_STATIC_DIR"])
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
async def schema() -> dict:
    node_registry = register_runner_nodes(NodeRegistry())
    type_registry = register_runner_types_for_ui(TypeRegistry())
    return export_ui_schema(node_registry, type_registry)


def dataset_response(item: Dataset, files: int) -> DatasetRead:
    return DatasetRead(id=item.id, name=item.name, files=files)


def voice_response(item: Voice) -> VoiceRead:
    return VoiceRead(id=item.id, name=item.name, segments=0, datasets=[])


@app.get("/datasets", response_model=list[DatasetRead])
async def list_datasets() -> list[DatasetRead]:
    with database_session() as session:
        rows = dataset_crud.list_dataset_file_counts(session)
        return [dataset_response(item, files) for item, files in rows]


@app.post("/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset(request: DatasetCreate) -> DatasetRead:
    try:
        with database_session() as session:
            item = dataset_crud.create_dataset(session, request)
            return dataset_response(item, 0)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(dataset_id: str) -> None:
    try:
        with database_session() as session:
            dataset_crud.delete_dataset(session, uuid.UUID(dataset_id))
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.get("/voices", response_model=VoicePage)
async def list_voices(
    query: str = "",
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> VoicePage:
    with database_session() as session:
        rows, total = voice_crud.search_voices(session, query, limit, offset)
        return VoicePage(rows=[voice_response(item) for item in rows], total=total)


@app.post("/voices", response_model=VoiceRead, status_code=status.HTTP_201_CREATED)
async def create_voice(request: VoiceCreate) -> VoiceRead:
    try:
        with database_session() as session:
            item = voice_crud.create_voice(session, request)
            return voice_response(item)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.patch("/voices/{voice_id}", response_model=VoiceRead)
async def rename_voice(voice_id: str, request: VoiceCreate) -> VoiceRead:
    try:
        with database_session() as session:
            item = voice_crud.rename_voice(session, uuid.UUID(voice_id), request.name)
            return voice_response(item)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.delete("/voices/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(voice_id: str) -> None:
    try:
        with database_session() as session:
            voice_crud.delete_voice(session, uuid.UUID(voice_id))
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.post("/graphs/runs", response_model=RunStatus, status_code=status.HTTP_202_ACCEPTED)
async def start_graph_run(request: InlineGraphRunRequest) -> RunStatus:
    try:
        logger.info("start graph run requested run_id=%s", request.run_id)
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


@app.get("/runs/{run_id}/graph", response_model=InlineGraphRunRequest)
async def get_run_graph(run_id: str) -> InlineGraphRunRequest:
    try:
        return await manager.graph(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@app.get("/runs/{run_id}/errors", response_model=list[RunEventResponse])
async def get_run_errors(run_id: str) -> list[RunEventResponse]:
    try:
        return await manager.errors(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.post("/runs/{run_id}/stop", response_model=RunStatus)
async def stop_run(run_id: str) -> RunStatus:
    try:
        logger.info("stop run requested run_id=%s", run_id)
        return await manager.stop(run_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@app.post("/runs/{run_id}/nodes/{node_id}/load", response_model=RunStatus)
async def load_run_node(run_id: str, node_id: str) -> RunStatus:
    try:
        logger.info("load node requested run_id=%s node_id=%s", run_id, node_id)
        return await manager.load_node(run_id, node_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@app.post("/runs/{run_id}/nodes/{node_id}/unload", response_model=RunStatus)
async def unload_run_node(run_id: str, node_id: str) -> RunStatus:
    try:
        logger.info("unload node requested run_id=%s node_id=%s", run_id, node_id)
        return await manager.unload_node(run_id, node_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@app.get("/runs/{run_id}/nodes/{node_id}/logs", response_model=NodeLogResponseMessage)
async def get_run_node_log(run_id: str, node_id: str) -> NodeLogResponseMessage:
    try:
        logger.info("node log requested run_id=%s node_id=%s", run_id, node_id)
        return await manager.node_log(run_id, node_id)
    except KeyError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@app.websocket("/ws")
async def backend_socket(websocket: WebSocket) -> None:
    try:
        logger.info("websocket connected")
        await manager.connect_socket(websocket)
    except WebSocketDisconnect:
        logger.info("websocket disconnected")
        return
