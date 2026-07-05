import json
from typing import Any

import nats
from nats.aio.client import Client as NatsClient
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError
from pydantic import BaseModel


DEFAULT_NATS_URL = "nats://127.0.0.1:4222"

COMMAND_STREAM = "RUNFLOW_COMMANDS"
EVENT_STREAM = "RUNFLOW_EVENTS"

COMMAND_SUBJECTS = ["runflow.commands.>"]
EVENT_SUBJECTS = ["runflow.events.*", "runflow.logs.*"]
RUNNER_HEARTBEAT_SUBJECT = "runflow.runners.heartbeat"

START_COMMAND_SUBJECT = "runflow.commands.start"
STOP_COMMAND_SUBJECT = "runflow.commands.stop"
NODE_COMMAND_SUBJECT = "runflow.commands.node"
NODE_LOG_COMMAND_SUBJECT = "runflow.commands.logs"
EVENT_SUBJECT_PREFIX = "runflow.events"
LOG_RESPONSE_SUBJECT_PREFIX = "runflow.logs"

RUNNER_COMMAND_DURABLE = "runflow-runners"
BACKEND_EVENT_DURABLE = "runflow-backend-events"
BACKEND_HEARTBEAT_DURABLE = "runflow-backend-runner-heartbeats"


async def connect(url: str = DEFAULT_NATS_URL) -> NatsClient:
    return await nats.connect(url)


def event_subject(run_id: str) -> str:
    return f"{EVENT_SUBJECT_PREFIX}.{run_id}"


def stop_command_subject(runner_id: str) -> str:
    return f"{STOP_COMMAND_SUBJECT}.{runner_id}"


def node_command_subject(runner_id: str) -> str:
    return f"{NODE_COMMAND_SUBJECT}.{runner_id}"


def node_log_command_subject(runner_id: str) -> str:
    return f"{NODE_LOG_COMMAND_SUBJECT}.{runner_id}"


def node_log_response_subject(request_id: str) -> str:
    return f"{LOG_RESPONSE_SUBJECT_PREFIX}.{request_id}"


def encode_model(model: BaseModel) -> bytes:
    return model.model_dump_json().encode("utf-8")


def decode_json(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JetStream payload must be a JSON object")
    return value


async def ensure_streams(js: JetStreamContext) -> None:
    await _ensure_stream(
        js,
        StreamConfig(
            name=COMMAND_STREAM,
            subjects=COMMAND_SUBJECTS,
            retention=RetentionPolicy.WORK_QUEUE,
            storage=StorageType.FILE,
        ),
    )
    await _ensure_stream(
        js,
        StreamConfig(
            name=EVENT_STREAM,
            subjects=[*EVENT_SUBJECTS, RUNNER_HEARTBEAT_SUBJECT],
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.FILE,
            max_age=60 * 60 * 6,
        ),
    )


async def _ensure_stream(js: JetStreamContext, config: StreamConfig) -> None:
    try:
        info = await js.stream_info(config.name)
    except NotFoundError:
        await js.add_stream(config=config)
        return
    if sorted(info.config.subjects) != sorted(config.subjects):
        await js.update_stream(config=config)
