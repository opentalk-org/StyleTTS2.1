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
  renderEvents();
}

function syncActiveRun(data) {
  if (state.activeRunId && data.runs.some((run) => run.run_id === state.activeRunId)) return;
  const active = data.runs.find((run) => ["queued", "running", "stopping"].includes(run.state));
  const fallback = data.runs[data.runs.length - 1];
  setActiveRun(active ? active.run_id : fallback?.run_id || null);
}

async function refreshEvents() {
  if (!state.activeRunId) {
    renderEvents();
    return;
  }
  const events = await json(`/runs/${encodeURIComponent(state.activeRunId)}/events?after=${state.eventAfter}`);
  if (events.length) {
    state.events.push(...events);
    state.eventAfter = events.at(-1).sequence;
    renderNodes();
    renderEdges();
  }
  renderEvents();
}

function nodeRunClass(nodeId) {
  const status = nodeMetrics(nodeId).status;
  return status ? ` run-${status}` : "";
}

function nodeRunPanel(nodeId) {
  const metrics = nodeMetrics(nodeId);
  const node = state.graph.nodes.find((item) => item.id === nodeId);
  if (node && nodeInfo(node.type).is_input) {
    return `<div class="node-metrics">
      <div class="node-metric node-metric-items"><strong>${metrics.remainingItems}</strong><span>items</span></div>
    </div>`;
  }
  return `<div class="node-metrics">
    <div class="node-metric node-metric-queue"><strong>${metrics.queued}</strong><span>queued</span></div>
  </div>`;
}

function nodeMetrics(nodeId) {
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

function renderEvents() {
  return null;
}

function lastFailure(nodeId) {
  return [...state.events].reverse().find((event) => event.node_id === nodeId && event.kind === "node_failed");
}
