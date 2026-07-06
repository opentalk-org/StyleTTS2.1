async function loadActiveRunGraph(runId = state.activeRunId, renderNow = true) {
  if (!runId) return false;
  try {
    const graph = await json(`/runs/${encodeURIComponent(runId)}/graph`);
    if (runId !== state.activeRunId) return false;
    applyRunGraph(graph, renderNow);
    state.loadedGraphRunId = runId;
    return true;
  } catch (error) {
    if (!String(error.message).includes("not available")) setLog(error.message);
    return false;
  }
}

function applyRunGraph(request, renderNow = true) {
  state.graph = {
    nodes: request.nodes.map((node) => ({
      id: node.id,
      type: node.type,
      x: Number(node.x || 0),
      y: Number(node.y || 0),
      params: structuredClone(node.params),
      runtime: structuredClone(node.runtime),
    })),
    edges: structuredClone(request.edges),
  };
  state.runtimeConfig = structuredClone(request.context.config);
  el.workDir.value = request.context.work_dir;
  el.outputDir.value = request.context.output_dir;
  const selected = selectedId();
  if (!state.graph.nodes.some((node) => node.id === selected)) {
    const first = state.graph.nodes[0];
    setSelection(first ? [first.id] : []);
  }
  if (renderNow) render();
}
