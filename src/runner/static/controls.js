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
  for (const button of el.tabs) button.classList.toggle("is-active", button.dataset.tab === state.rightTab);
  for (const panel of el.tabPanels) panel.classList.toggle("is-active", panel.dataset.panel === state.rightTab);
  renderSettings();
  renderNodeRuntime();
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
    nodes: state.graph.nodes.map((node) => ({ id: node.id, type: node.type, params: node.params, runtime: node.runtime })),
    edges: state.graph.edges,
    context: { work_dir: el.workDir.value, output_dir: el.outputDir.value, config: state.runtimeConfig },
  };
}

async function runGraph() {
  try {
    const result = await json("/graphs/runs", { method: "POST", body: JSON.stringify(graphPayload()) });
    setActiveRun(result.run_id);
    setLog(`started ${result.run_id}`);
    await refreshRuns();
  } catch (error) {
    setLog(error.message);
  }
}

async function refreshRuns() {
  try {
    const data = await json("/runs");
    el.api.textContent = "online";
    el.api.className = "status status--ok";
    el.active.textContent = `${data.active_runs} active`;
    syncActiveRun(data);
    el.runs.innerHTML = data.runs.map(runHtml).join("") || `<span class="empty">No runs yet.</span>`;
    bindStopButtons();
    await refreshEvents();
  } catch (error) {
    el.api.textContent = "offline";
    el.api.className = "status status--off";
    setLog(error.message);
  }
}

function runHtml(run) {
  const stop = ["queued", "running", "stopping"].includes(run.state) ? `<button class="btn" data-stop="${run.run_id}">Stop</button>` : "<span></span>";
  return `<article class="run">
    <i class="run-led state-${run.state}"></i>
    <div class="run-body" data-run="${run.run_id}"><span class="run-id">${run.run_id}</span><span class="run-meta">${run.state} · ${run.workflow_path} · ${run.event_count} events</span></div>
    ${stop}
  </article>`;
}

function bindStopButtons() {
  for (const button of el.runs.querySelectorAll("[data-stop]")) {
    button.addEventListener("click", async () => {
      await json(`/runs/${encodeURIComponent(button.dataset.stop)}/stop`, { method: "POST" });
      await refreshRuns();
    });
  }
  for (const row of el.runs.querySelectorAll("[data-run]")) {
    row.addEventListener("click", async () => {
      setActiveRun(row.dataset.run);
      await refreshEvents();
    });
  }
}

function onKeydown(event) {
  if (event.key === "Escape") {
    if (state.wire) state.wire = null;
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
  await refreshRuns();
  window.setInterval(refreshRuns, 300);
}

el.nodes.addEventListener("pointerdown", onNodesPointerDown);
el.nodes.addEventListener("click", onNodesClick);
el.canvas.addEventListener("pointerdown", onCanvasPointerDown);
el.template.addEventListener("click", loadTemplate);
el.run.addEventListener("click", runGraph);
el.refresh.addEventListener("click", refreshRuns);
el.clear.addEventListener("click", clearGraph);
window.addEventListener("resize", renderEdges);
window.addEventListener("keydown", onKeydown);
for (const button of el.tabs) {
  button.addEventListener("click", () => {
    state.rightTab = button.dataset.tab;
    renderInspector();
  });
}
init().catch((error) => setLog(error.message));
