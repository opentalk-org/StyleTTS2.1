import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from runner.graphs import build_inline_graph
from shared.db.workflows.schemas import WorkflowDefinition, WorkflowLaunchSource, WorkflowRead
from shared.schemas import GraphNodeRequest, InlineGraphRunRequest


SOURCE_NODE_TYPES = {
    "selected_audio": "AudioSource",
    "dataset_audio": "AudioSource",
    "all_audio": "AudioSource",
}


def _examples_dir() -> Path:
    override = os.environ.get("RUNFLOW_WORKFLOWS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "workflows"


def load_example_workflows() -> list[WorkflowRead]:
    """Read the ``workflows/*.json`` example definitions fresh from disk.

    Called per request so edits to the folder show up without a backend or
    database restart. Files that are missing, malformed, or not valid workflow
    definitions are skipped rather than failing the whole listing.
    """
    examples: list[WorkflowRead] = []
    for path in sorted(_examples_dir().glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            definition = WorkflowDefinition.model_validate(payload["data"])
        except (OSError, KeyError, TypeError, ValueError, ValidationError):
            continue
        examples.append(
            WorkflowRead(
                id=uuid5(NAMESPACE_URL, f"workflow-example:{path.name}"),
                name=str(payload.get("name") or path.stem),
                data=definition,
                hidden=bool(payload.get("hidden", False)),
            )
        )
    return examples


def compile_workflow_definition(
    definition: WorkflowDefinition,
    workflow_id: UUID,
    run_id: str | None = None,
) -> InlineGraphRunRequest:
    request = InlineGraphRunRequest(
        run_id=run_id,
        nodes=[node.model_copy(deep=True) for node in definition.nodes],
        edges=definition.edges,
        context=definition.context,
    )
    if definition.launch_source is not None:
        request.nodes = _apply_launch_source(request.nodes, definition.launch_source)
    build_inline_graph(request)
    return request.model_copy(update={"run_id": request.run_id or f"workflow_{workflow_id.hex[:8]}"})


def _apply_launch_source(
    nodes: list[GraphNodeRequest],
    launch_source: WorkflowLaunchSource,
) -> list[GraphNodeRequest]:
    source_type = SOURCE_NODE_TYPES[launch_source.kind]
    patched = []
    matched = False
    for node in nodes:
        if node.type == source_type:
            patched.append(node.model_copy(update={"params": _source_params(launch_source)}))
            matched = True
        else:
            patched.append(node)
    if not matched:
        raise ValueError(f"Workflow has no source node for launch source: {launch_source.kind}")
    return patched


def _source_params(launch_source: WorkflowLaunchSource) -> dict:
    if launch_source.kind == "selected_audio":
        return {"source": "selected", "audio_file_ids": [str(audio_id) for audio_id in launch_source.audio_file_ids], "include_virtual": launch_source.include_virtual}
    if launch_source.kind == "dataset_audio":
        if launch_source.dataset_id is None:
            raise ValueError("dataset_audio launch source requires dataset_id")
        return {"source": "dataset", "dataset_id": str(launch_source.dataset_id), "include_virtual": launch_source.include_virtual}
    if launch_source.kind == "all_audio":
        return {"source": "all", "include_virtual": launch_source.include_virtual}
    raise ValueError(f"Unknown launch source: {launch_source.kind}")
