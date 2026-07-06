function renderSettings() {
  if (state.selection.size > 1) {
    el.settings.innerHTML = `<p class="empty">${state.selection.size} nodes selected. Drag to move the group, Delete to remove.</p>`;
    return;
  }
  const node = state.graph.nodes.find((item) => item.id === selectedId());
  if (!node) {
    el.settings.innerHTML = `<p class="empty">Select a node to edit its settings.</p>`;
    return;
  }
  const info = nodeInfo(node.type);
  el.settings.innerHTML = `<label class="field">node id<input id="nodeIdInput" value="${node.id}" spellcheck="false"></label>`;
  document.querySelector("#nodeIdInput").addEventListener("change", (event) => renameNode(node, event.target.value));
  const fields = document.createElement("div");
  el.settings.appendChild(fields);
  renderSchemaForm(fields, info.settings, node.params, () => renderSettings());
}

function renderNodeRuntime() {
  if (state.selection.size > 1) {
    el.runtimeSettings.innerHTML = `<p class="empty">${state.selection.size} nodes selected.</p>`;
    return;
  }
  const node = state.graph.nodes.find((item) => item.id === selectedId());
  if (!node) {
    el.runtimeSettings.innerHTML = `<p class="empty">Select a node to edit runtime config.</p>`;
    return;
  }
  const info = nodeInfo(node.type);
  renderSchemaForm(el.runtimeSettings, info.runtime, node.runtime, () => renderNodeRuntime());
}

function renderInspector() {
  el.popup.hidden = !state.inspectorOpen;
  for (const button of el.tabs) button.classList.toggle("is-active", button.dataset.tab === state.rightTab);
  for (const panel of el.tabPanels) panel.classList.toggle("is-active", panel.dataset.panel === state.rightTab);
  renderSettings();
  renderNodeRuntime();
  renderNodeLogs();
}

function openInspector(tab = "settings") {
  state.rightTab = tab;
  state.inspectorOpen = true;
  renderInspector();
  if (tab === "logs") loadSelectedNodeLog();
}

function closeInspector() {
  state.inspectorOpen = false;
  renderInspector();
}

function toggleContextPanel() {
  el.contextPanel.hidden = !el.contextPanel.hidden;
}

function renderContextSettings() {
  renderSchemaForm(el.contextSettings, state.schema.runtime_config, state.runtimeConfig, () => renderContextSettings());
}

function renameNode(node, nextId) {
  const previous = node.id;
  node.id = nextId;
  for (const edge of state.graph.edges) {
    if (edge.source_node === previous) edge.source_node = nextId;
    if (edge.target_node === previous) edge.target_node = nextId;
  }
  setSelection([nextId]);
  render();
}

function graphPayload() {
  return {
    run_id: el.runId.value || null,
    nodes: state.graph.nodes.map((node) => ({ id: node.id, type: node.type, x: node.x, y: node.y, params: node.params, runtime: node.runtime })),
    edges: state.graph.edges,
    context: { work_dir: el.workDir.value, output_dir: el.outputDir.value, config: state.runtimeConfig },
  };
}

async function runGraph() {
  try {
    const result = await json("/graphs/runs", { method: "POST", body: JSON.stringify(graphPayload()) });
    applyRunStatus(result);
    setActiveRun(result.run_id);
    setLog(`started ${result.run_id}`);
    await loadActiveRunState();
  } catch (error) {
    setLog(error.message);
  }
}

function activeRunStatus() {
  if (!state.activeRunId) return null;
  return state.runs.find((run) => run.run_id === state.activeRunId) || null;
}

async function stopActiveRun() {
  const run = activeRunStatus();
  if (!run || !isActiveRunState(run.state)) return;
  try {
    const result = await json(`/runs/${encodeURIComponent(run.run_id)}/stop`, { method: "POST" });
    applyRunStatus(result);
    setLog(`stop requested for ${run.run_id}`);
  } catch (error) {
    setLog(error.message);
  }
}

async function onRunButtonClick() {
  const run = activeRunStatus();
  if (run && isActiveRunState(run.state)) {
    await stopActiveRun();
    return;
  }
  await runGraph();
}

function renderRunButton() {
  const run = activeRunStatus();
  const stopping = run && run.state === "stopping";
  const stoppingDisabled = Boolean(stopping);
  const canStop = run && isActiveRunState(run.state);
  el.run.textContent = stopping ? "Stopping..." : canStop ? "Stop run" : "Run graph";
  el.run.disabled = stoppingDisabled;
  el.run.classList.toggle("btn-stop", Boolean(canStop));
}

async function controlNodeLifecycle(nodeId, action) {
  if (!state.activeRunId) {
    setLog("start or select a run before controlling node lifecycle");
    return;
  }
  try {
    const runId = encodeURIComponent(state.activeRunId);
    const node = encodeURIComponent(nodeId);
    const result = await json(`/runs/${runId}/nodes/${node}/${action}`, { method: "POST" });
    applyRunStatus(result);
    setLog(`${action} requested for ${nodeId}`);
  } catch (error) {
    setLog(error.message);
  }
}

async function refreshRuns() {
  try {
    const data = await json("/runs");
    el.api.textContent = "online";
    el.api.className = "status status--ok";
    applyRunnerStatus(data);
    await loadActiveRunState();
  } catch (error) {
    el.api.textContent = "offline";
    el.api.className = "status status--off";
    setLog(error.message);
  }
}

function renderRuns() {
  el.runs.innerHTML = state.runs.map(runHtml).join("") || `<span class="empty">No runs yet.</span>`;
  bindRunRows();
  renderRunButton();
}

function runHtml(run) {
  return `<article class="run">
    <i class="run-led state-${run.state}"></i>
    <div class="run-body" data-run="${run.run_id}"><span class="run-id">${run.run_id}</span><span class="run-meta">${run.state} · ${run.workflow_path} · ${run.event_count} events</span></div>
  </article>`;
}

function bindRunRows() {
  for (const row of el.runs.querySelectorAll("[data-run]")) {
    row.addEventListener("click", async () => {
      setActiveRun(row.dataset.run);
      await loadActiveRunState();
    });
  }
}

function onKeydown(event) {
  if (event.key === "Escape") {
    if (state.inspectorOpen) closeInspector();
    else if (state.wire) state.wire = null;
    else state.selection = new Set();
    dragLoop.onUp = null;
    render();
  }
  if ((event.key === "Delete" || event.key === "Backspace") && state.selection.size) {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    deleteSelection();
  }
}

async function init() {
  state.schema = await json("/schema");
  state.runtimeConfig = structuredClone(state.schema.runtime_config_defaults);
  renderPalette();
  renderLegend();
  renderContextSettings();
  loadTemplate();
  connectBackendSocket();
  await refreshRuns();
}

el.nodes.addEventListener("pointerdown", onNodesPointerDown);
el.nodes.addEventListener("click", onNodesClick);
el.canvas.addEventListener("pointerdown", onCanvasPointerDown);
el.canvas.addEventListener("wheel", onCanvasWheel, { passive: false });
el.template.addEventListener("click", loadTemplate);
el.run.addEventListener("click", onRunButtonClick);
el.refresh.addEventListener("click", refreshRuns);
el.clear.addEventListener("click", clearGraph);
el.zoomIn.addEventListener("click", () => zoomBy(1.2));
el.zoomOut.addEventListener("click", () => zoomBy(0.8));
el.zoomReset.addEventListener("click", () => setZoom(1));
el.context.addEventListener("click", toggleContextPanel);
el.popupClose.addEventListener("click", closeInspector);
el.popup.addEventListener("click", (event) => {
  if (event.target.closest("[data-close-popup]")) closeInspector();
});
window.addEventListener("resize", renderEdges);
window.addEventListener("keydown", onKeydown);
for (const button of el.tabs) {
  button.addEventListener("click", () => {
    state.rightTab = button.dataset.tab;
    renderInspector();
    if (state.rightTab === "logs") loadSelectedNodeLog();
  });
}
init().catch((error) => setLog(error.message));
