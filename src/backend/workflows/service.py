from uuid import UUID

from runner.graphs import build_inline_graph
from shared.db.workflows.schemas import WorkflowDefinition, WorkflowLaunchSource
from shared.schemas import GraphNodeRequest, InlineGraphRunRequest


SOURCE_NODE_TYPES = {
    "selected_audio": "SelectedAudioSource",
    "dataset_audio": "DatasetAudioSource",
    "all_audio": "AllAudioSource",
}


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
        return {"audio_file_ids": [str(audio_id) for audio_id in launch_source.audio_file_ids]}
    if launch_source.kind == "dataset_audio":
        if launch_source.dataset_id is None:
            raise ValueError("dataset_audio launch source requires dataset_id")
        return {"dataset_id": str(launch_source.dataset_id), "include_virtual": launch_source.include_virtual}
    if launch_source.kind == "all_audio":
        return {"include_virtual": launch_source.include_virtual}
    raise ValueError(f"Unknown launch source: {launch_source.kind}")
