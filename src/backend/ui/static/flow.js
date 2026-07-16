function html(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setActiveRun(runId) {
  if (state.activeRunId === runId) return;
  state.activeRunId = runId;
  state.activeRunLoadToken += 1;
  state.loadedGraphRunId = null;
  state.runSnapshot = null;
  sendRunWatch();
  if (typeof renderRunButton === "function") renderRunButton();
}

function isActiveRunState(stateName) {
  return ["queued", "running", "stopping"].includes(stateName);
}

function syncActiveRun(data) {
  if (state.activeRunId && data.runs.some((run) => run.run_id === state.activeRunId)) return;
  const active = data.runs.find((run) => isActiveRunState(run.state));
  const fallback = data.runs[data.runs.length - 1];
  setActiveRun(active ? active.run_id : fallback?.run_id || null);
}

async function loadActiveRunState() {
  const runId = state.activeRunId;
  const token = ++state.activeRunLoadToken;
  state.activeRunLoading = true;
  if (!runId) {
    state.runSnapshot = null;
    state.activeRunLoading = false;
    return;
  }
  try {
    await loadActiveRunGraph(runId, false);
    if (token !== state.activeRunLoadToken || runId !== state.activeRunId) return;
    state.runSnapshot = await json(`/runs/${encodeURIComponent(runId)}/snapshot`);
  } finally {
    if (token === state.activeRunLoadToken) state.activeRunLoading = false;
  }
  renderNodes();
  renderEdges();
  refreshOpenNodeLogs();
}

function connectBackendSocket() {
  if (state.backendSocket) return;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
  state.backendSocket = socket;
  socket.addEventListener("open", () => {
    el.api.textContent = "online";
    el.api.className = "status status--ok";
    sendRunWatch();
  });
  socket.addEventListener("message", (event) => applyBackendSocketMessage(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    if (state.backendSocket === socket) {
      state.backendSocket = null;
      el.api.textContent = "offline";
      el.api.className = "status status--off";
    }
  });
}

function sendRunWatch() {
  if (!state.backendSocket || state.backendSocket.readyState !== WebSocket.OPEN) return;
  state.backendSocket.send(JSON.stringify({ type: "watch_run", run_id: state.activeRunId }));
}

function applyBackendSocketMessage(message) {
  if (message.type === "runner_status") {
    applyRunnerStatus(message.status);
    return;
  }
  if (message.status && message.type !== "runner_status") applyRunStatus(message.status);
  const messageRunId = message.run_id || (message.event ? message.event.run_id : null);
  if (messageRunId === state.activeRunId && state.loadedGraphRunId !== state.activeRunId) {
    if (state.activeRunLoading) return;
    loadActiveRunState().catch((error) => setLog(error.message));
    return;
  }
  if (message.snapshot && messageRunId === state.activeRunId) state.runSnapshot = message.snapshot;
  renderNodes();
  renderEdges();
  if (state.inspectorOpen && state.rightTab === "logs") renderNodeLogs();
}

function applyRunnerStatus(data) {
  const previousRunId = state.activeRunId;
  state.runs = data.runs;
  el.active.textContent = `${data.active_runs} active`;
  syncActiveRun(data);
  if (typeof renderRunButton === "function") renderRunButton();
  if (state.activeRunId && state.activeRunId !== previousRunId) {
    loadActiveRunState().catch((error) => setLog(error.message));
  }
  renderRuns();
}

function applyRunStatus(run) {
  const index = state.runs.findIndex((item) => item.run_id === run.run_id);
  if (index === -1) state.runs.push(run);
  else state.runs[index] = run;
  const activeRuns = state.runs.filter((item) => isActiveRunState(item.state));
  el.active.textContent = `${activeRuns.length} active`;
  if (typeof renderRunButton === "function") renderRunButton();
  renderRuns();
}

function nodeRunClass(nodeId) {
  const status = nodeMetrics(nodeId).status;
  return status ? ` run-${status}` : "";
}

function nodeRunPanel(nodeId) {
  const metrics = nodeMetrics(nodeId);
  const node = state.graph.nodes.find((item) => item.id === nodeId);
  const open = `<button class="node-open" type="button" aria-label="Open node inspector">edit</button>`;
  const loadState = metrics.loaded ? "loaded" : "unloaded";
  const runState = metrics.status === "running" ? "running" : "idle";
  const lifecycle = `<div class="node-lifecycle">
    <span class="node-load ${loadState}">${loadState}</span>
    <span class="node-run-state ${runState}">${runState}</span>
    <button class="node-load-action" type="button" data-node-load="${nodeId}">load</button>
    <button class="node-load-action" type="button" data-node-unload="${nodeId}">unload</button>
  </div>`;
  if (node && nodeInfo(node.type).is_input) {
    return `<div class="node-metrics">
      <div class="node-metric node-metric-items"><strong>${metrics.remainingItems}</strong><span>left</span></div>
      <div class="node-metric node-metric-state ${metrics.status}"><strong>${metrics.completed}</strong><span>done</span></div>
      ${lifecycle}
      ${open}
    </div>`;
  }
  return `<div class="node-metrics">
    <div class="node-metric node-metric-queue"><strong>${metrics.queued}</strong><span>queued</span></div>
    <div class="node-metric node-metric-state ${metrics.status}"><strong>${metrics.completed}</strong><span>done</span></div>
    ${lifecycle}
    ${open}
  </div>`;
}

function nodeMetrics(nodeId) {
  const snapshot = nodeSnapshot(nodeId);
  if (!snapshot) return { queued: 0, remainingItems: 0, running: 0, completed: 0, received: 0, failed: false, loaded: false, status: "idle" };
  const discovered = snapshot.counters.input_items_discovered;
  const completed = discovered === undefined
    ? snapshot.counters.items_completed || snapshot.counters.tasks_completed || 0
    : Math.min(discovered, snapshot.counters.packets_created || Math.max(0, discovered - (snapshot.remaining_items || 0)));
  const remainingItems = discovered === undefined ? snapshot.remaining_items || 0 : discovered - completed;
  return {
    queued: snapshot.queue_size,
    remainingItems,
    running: snapshot.running_batches,
    completed,
    received: snapshot.counters.packets_received || 0,
    failed: snapshot.status === "failed",
    loaded: snapshot.loaded,
    status: snapshot.status,
  };
}

function nodeSnapshot(nodeId) {
  if (!state.runSnapshot) return null;
  return state.runSnapshot.nodes.find((node) => node.node_id === nodeId) || null;
}

function lastFailure(nodeId) {
  const snapshot = nodeSnapshot(nodeId);
  if (snapshot && snapshot.error) return { message: snapshot.error };
  return null;
}
