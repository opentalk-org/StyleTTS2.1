function nodeLogKey(runId, nodeId) {
  return `${runId}:${nodeId}`;
}

async function loadSelectedNodeLog() {
  const nodeId = selectedId();
  if (!state.activeRunId || !nodeId) return;
  const key = nodeLogKey(state.activeRunId, nodeId);
  if (state.nodeLogLoading.has(key)) return;
  state.nodeLogLoading.add(key);
  renderNodeLogs();
  try {
    const runId = encodeURIComponent(state.activeRunId);
    const node = encodeURIComponent(nodeId);
    state.nodeLogs[key] = await json(`/runs/${runId}/nodes/${node}/logs`);
  } catch (error) {
    state.nodeLogs[key] = { content: "", truncated: false, error: error.message };
  } finally {
    state.nodeLogLoading.delete(key);
    renderNodeLogs();
  }
}

function renderNodeLogs() {
  if (!el.nodeLogs) return;
  const nodeId = selectedId();
  if (!nodeId) {
    el.nodeLogs.innerHTML = `<p class="empty">Select one node to view logs.</p>`;
    return;
  }
  if (!state.activeRunId) {
    el.nodeLogs.innerHTML = `<p class="empty">Start or select a run to view node logs.</p>`;
    return;
  }
  const key = nodeLogKey(state.activeRunId, nodeId);
  const cached = state.nodeLogs[key];
  if (!cached && !state.nodeLogLoading.has(key)) loadSelectedNodeLog();
  if (state.nodeLogLoading.has(key)) {
    el.nodeLogs.innerHTML = `<p class="empty">Loading logs for ${html(nodeId)}.</p>`;
    return;
  }
  if (!cached) {
    el.nodeLogs.innerHTML = `<p class="empty">No logs loaded.</p>`;
    return;
  }
  const warning = cached.truncated ? `<p class="log-note">Showing latest 1 MB.</p>` : "";
  const error = cached.error ? `<p class="log-error">${html(cached.error)}</p>` : "";
  const content = cached.content || "No log lines for this node yet.";
  el.nodeLogs.innerHTML = `${warning}${error}<pre class="node-log-body">${html(content)}</pre>`;
}

function refreshOpenNodeLogs() {
  if (state.inspectorOpen && state.rightTab === "logs") loadSelectedNodeLog();
}
