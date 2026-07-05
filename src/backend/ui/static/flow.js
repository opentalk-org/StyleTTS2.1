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
  state.events = [];
  state.eventAfter = 0;
  state.runSnapshot = null;
  renderEvents();
}

function syncActiveRun(data) {
  if (state.activeRunId && data.runs.some((run) => run.run_id === state.activeRunId)) return;
  const active = data.runs.find((run) => ["queued", "running", "stopping"].includes(run.state));
  const fallback = data.runs[data.runs.length - 1];
  setActiveRun(active ? active.run_id : fallback?.run_id || null);
}

async function loadActiveRunState() {
  if (!state.activeRunId) {
    state.runSnapshot = null;
    renderEvents();
    return;
  }
  state.runSnapshot = await json(`/runs/${encodeURIComponent(state.activeRunId)}/snapshot`);
  const events = await json(`/runs/${encodeURIComponent(state.activeRunId)}/events?after=${state.eventAfter}`);
  if (events.length) {
    state.events.push(...events);
    if (state.events.length > 1000) state.events = state.events.slice(-1000);
    state.eventAfter = events.at(-1).sequence;
  }
  renderNodes();
  renderEdges();
  renderEvents();
}

function connectBackendSocket() {
  if (state.backendSocket) return;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
  state.backendSocket = socket;
  socket.addEventListener("open", () => {
    el.api.textContent = "online";
    el.api.className = "status status--ok";
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

function applyBackendSocketMessage(message) {
  if (message.type === "runner_status") applyRunnerStatus(message.status);
  if (message.status && message.type !== "runner_status") applyRunStatus(message.status);
  const messageRunId = message.event ? message.event.run_id : null;
  if (message.snapshot && messageRunId === state.activeRunId) state.runSnapshot = message.snapshot;
  if (message.event && message.event.run_id === state.activeRunId) {
    state.events.push(message.event);
    if (state.events.length > 1000) state.events = state.events.slice(-1000);
    state.eventAfter = Math.max(state.eventAfter, message.event.sequence);
  }
  renderNodes();
  renderEdges();
  renderEvents();
}

function applyRunnerStatus(data) {
  const previousRunId = state.activeRunId;
  state.runs = data.runs;
  el.active.textContent = `${data.active_runs} active`;
  syncActiveRun(data);
  if (state.activeRunId && state.activeRunId !== previousRunId) {
    loadActiveRunState().catch((error) => setLog(error.message));
  }
  renderRuns();
}

function applyRunStatus(run) {
  const index = state.runs.findIndex((item) => item.run_id === run.run_id);
  if (index === -1) state.runs.push(run);
  else state.runs[index] = run;
  const activeRuns = state.runs.filter((item) => ["queued", "running", "stopping"].includes(item.state));
  el.active.textContent = `${activeRuns.length} active`;
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
  if (node && nodeInfo(node.type).is_input) {
    return `<div class="node-metrics">
      <div class="node-metric node-metric-items"><strong>${metrics.remainingItems}</strong><span>items</span></div>
      ${open}
    </div>`;
  }
  return `<div class="node-metrics">
    <div class="node-metric node-metric-queue"><strong>${metrics.queued}</strong><span>queued</span></div>
    ${open}
  </div>`;
}

function nodeMetrics(nodeId) {
  const snapshot = nodeSnapshot(nodeId);
  if (snapshot) {
    return {
      queued: snapshot.queue_size,
      remainingItems: snapshot.remaining_items || 0,
      running: snapshot.running_batches,
      failed: snapshot.status === "failed",
      loaded: snapshot.loaded,
      status: snapshot.status === "failed" ? "failed" : snapshot.running_batches > 0 ? "running" : "idle",
    };
  }
  const metrics = { queued: 0, remainingItems: 0, running: 0, failed: false, loaded: false };
  for (const event of state.events) {
    if (event.node_id !== nodeId) continue;
    if (event.kind === "node_loaded") metrics.loaded = true;
    if (event.kind === "node_unloaded") metrics.loaded = false;
    if (event.kind === "input_items_discovered") metrics.remainingItems = Number(event.detail.item_count || 0);
    if (event.kind === "input_items_remaining") metrics.remainingItems = Number(event.detail.item_count || 0);
    if (event.kind === "task_enqueued" || event.kind === "queue_depth") metrics.queued = Number(event.detail.queue_size || 0);
    if (event.kind === "batch_started") {
      metrics.running += 1;
    }
    if (event.kind === "batch_completed") {
      metrics.running = Math.max(0, metrics.running - 1);
    }
    if (event.kind === "node_failed") {
      metrics.running = 0;
      metrics.failed = true;
    }
  }
  metrics.status = metrics.failed ? "failed" : metrics.running > 0 ? "running" : "idle";
  return metrics;
}

function nodeSnapshot(nodeId) {
  if (!state.runSnapshot) return null;
  return state.runSnapshot.nodes.find((node) => node.node_id === nodeId) || null;
}

function renderEvents() {
  return null;
}

function lastFailure(nodeId) {
  const snapshot = nodeSnapshot(nodeId);
  if (snapshot && snapshot.error) return { message: snapshot.error };
  return [...state.events].reverse().find((event) => event.node_id === nodeId && event.kind === "node_failed");
}
