async function json(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function setLog(message) {
  el.log.textContent = message || "";
}

function nodeInfo(type) {
  return state.schema.nodes[type];
}

function typeColor(typeName) {
  const type = state.schema.types[typeName];
  return type ? type.color : "var(--accent)";
}

/* ---- selection ------------------------------------------------------ */
function setSelection(ids) {
  state.selection = new Set(ids);
}

function toggleSelection(id) {
  if (state.selection.has(id)) state.selection.delete(id);
  else state.selection.add(id);
}

function selectedId() {
  return state.selection.size === 1 ? [...state.selection][0] : null;
}

/* ---- graph mutations ------------------------------------------------ */
function addNode(type, x = 130, y = 120) {
  const info = nodeInfo(type);
  const node = { id: `${type}_${state.seq++}`, type, x: x - state.pan.x, y: y - state.pan.y, params: structuredClone(info.settings_defaults) };
  state.graph.nodes.push(node);
  setSelection([node.id]);
  render();
}

function deleteNode(id) {
  state.graph.nodes = state.graph.nodes.filter((node) => node.id !== id);
  state.graph.edges = state.graph.edges.filter((edge) => edge.source_node !== id && edge.target_node !== id);
  state.selection.delete(id);
  render();
}

function deleteSelection() {
  const ids = state.selection;
  state.graph.nodes = state.graph.nodes.filter((node) => !ids.has(node.id));
  state.graph.edges = state.graph.edges.filter((edge) => !ids.has(edge.source_node) && !ids.has(edge.target_node));
  state.selection = new Set();
  render();
}

function clearGraph() {
  state.graph = { nodes: [], edges: [] };
  state.selection = new Set();
  state.wire = null;
  render();
}

function portType(nodeId, portName, kind) {
  const node = state.graph.nodes.find((item) => item.id === nodeId);
  const ports = kind === "output" ? nodeInfo(node.type).outputs : nodeInfo(node.type).inputs;
  const port = Object.values(ports).find((item) => item.name === portName);
  return port ? port.type : null;
}

// Mirror of runflow.core.types.dtype_accepts: a union input accepts a source
// only if every source member is one of its members; a union source fans out
// to every member. Types here carry `members` (empty for concrete types).
function typeAccepts(targetType, sourceType) {
  const target = state.schema.types[targetType];
  const source = state.schema.types[sourceType];
  if (target.members.length) {
    if (source.members.length) return source.members.every((member) => target.members.includes(member));
    return target.members.includes(sourceType);
  }
  if (source.members.length) return source.members.every((member) => typeAccepts(targetType, member));
  return targetType === sourceType;
}

function connect(sourceNode, sourcePort, targetNode, targetPort) {
  if (sourceNode === targetNode) return render();
  const sourceType = portType(sourceNode, sourcePort, "output");
  const targetType = portType(targetNode, targetPort, "input");
  if (!typeAccepts(targetType, sourceType)) {
    setLog(`can't wire ${sourceType} → ${targetType}: signal types don't match`);
    return render();
  }
  // one edge per input socket: a fresh wire replaces whatever fed it
  state.graph.edges = state.graph.edges.filter((edge) => !(edge.target_node === targetNode && edge.target_port === targetPort));
  state.graph.edges.push({ source_node: sourceNode, source_port: sourcePort, target_node: targetNode, target_port: targetPort });
  render();
}

function loadTemplate() {
  state.pan = { x: 0, y: 0 };
  state.graph = {
    nodes: [
      { id: "input", type: "DirectoryInput", x: 40, y: 130, params: { directory: ".", patterns: ["pyproject.toml"], repeat_count: 80, sleep_sec: 0.01 } },
      { id: "load_audio", type: "LoadAudio", x: 320, y: 110, params: { sample_rate: 16000, channels: 1, sleep_sec: 0.05 } },
      { id: "vad", type: "VADDetect", x: 570, y: 20, params: { max_segment_sec: 30, padding_sec: 0.1, sleep_sec: 0.05 } },
      { id: "cut_segments", type: "AudioCutBySegments", x: 570, y: 210, params: { sleep_sec: 0.03 } },
      { id: "whisper", type: "Whisper", x: 830, y: 180, params: { language: "auto", sleep_sec: 0.15 } },
      { id: "save_transcript", type: "SaveTranscript", x: 1090, y: 130, params: { output_dir: null, sleep_sec: 0.02 } },
      { id: "save_audio", type: "SaveAudio", x: 1090, y: 320, params: { output_dir: null, sleep_sec: 0.02 } },
    ],
    edges: [
      { source_node: "input", source_port: "paths", target_node: "load_audio", target_port: "path" },
      { source_node: "load_audio", source_port: "audio", target_node: "vad", target_port: "audio" },
      { source_node: "load_audio", source_port: "audio", target_node: "cut_segments", target_port: "audio" },
      { source_node: "vad", source_port: "segments", target_node: "cut_segments", target_port: "segments" },
      { source_node: "cut_segments", source_port: "chunks", target_node: "whisper", target_port: "audio" },
      { source_node: "cut_segments", source_port: "chunks", target_node: "save_audio", target_port: "audio" },
      { source_node: "whisper", source_port: "transcript", target_node: "save_transcript", target_port: "transcript" },
    ],
  };
  setSelection(["input"]);
  state.seq = 1;
  state.wire = null;
  render();
}

/* ---- rails ---------------------------------------------------------- */
function renderPalette() {
  const groups = new Map();
  for (const item of Object.values(state.schema.nodes)) {
    if (!groups.has(item.category)) groups.set(item.category, []);
    groups.get(item.category).push(item);
  }
  el.palette.innerHTML = "";
  for (const category of [...groups.keys()].sort()) {
    const group = document.createElement("div");
    group.className = "palette-group";
    group.innerHTML = `<p class="palette-cat">${category}</p>`;
    for (const item of groups.get(category).sort((a, b) => a.type.localeCompare(b.type))) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "palette-item";
      button.innerHTML = `<i class="dot" style="color:${nodeAccent(item)};background:${nodeAccent(item)}"></i>${item.type}`;
      button.addEventListener("click", () => addNode(item.type));
      group.appendChild(button);
    }
    el.palette.appendChild(group);
  }
}

function renderLegend() {
  el.legend.innerHTML = Object.values(state.schema.types)
    .map((type) => `<span class="legend-item"><i class="dot" style="color:${type.color};background:${type.color}"></i>${type.name}</span>`)
    .join("");
}

/* ---- canvas render -------------------------------------------------- */
function render() {
  renderNodes();
  renderEdges();
  renderSettings();
  el.canvas.classList.toggle("has-nodes", state.graph.nodes.length > 0);
}

function applyPan() {
  el.nodes.style.transform = `translate(${state.pan.x}px, ${state.pan.y}px)`;
}

function renderNodes() {
  applyPan();
  el.nodes.innerHTML = "";
  for (const node of state.graph.nodes) {
    const info = nodeInfo(node.type);
    const card = document.createElement("article");
    card.className = `${state.selection.has(node.id) ? "node selected" : "node"}${nodeRunClass(node.id)}`;
    card.style.transform = `translate(${node.x}px, ${node.y}px)`;
    card.dataset.id = node.id;
    card.innerHTML = `
      <div class="node-title"><div class="node-name"><strong>${node.id}</strong><span class="type">${node.type}</span></div></div>
      <button class="node-del" type="button" aria-label="Delete node">&times;</button>
      ${nodeRunPanel(node.id)}
      <div class="ports">
        <div class="inputs">${portsHtml(node, info.inputs, "input")}</div>
        <div class="outputs">${portsHtml(node, info.outputs, "output")}</div>
      </div>`;
    el.nodes.appendChild(card);
  }
}

function portsHtml(node, ports, kind) {
  return Object.values(ports)
    .map((port) => {
      const socket = `<i class="socket" style="--sock:${typeColor(port.type)}" data-node="${node.id}" data-port="${port.name}" data-kind="${kind}"></i>`;
      return `<div class="port">${kind === "input" ? socket : ""}<span>${port.name}</span>${kind === "output" ? socket : ""}</div>`;
    })
    .join("");
}

function edgeColor(sourceNode, sourcePort) {
  const info = nodeInfo(state.graph.nodes.find((node) => node.id === sourceNode)?.type);
  const port = info && Object.values(info.outputs).find((item) => item.name === sourcePort);
  return port ? typeColor(port.type) : "var(--accent)";
}

function cablePath(start, end) {
  const dx = Math.max(40, Math.abs(end.x - start.x) * 0.5);
  return `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
}

function renderEdges() {
  const box = el.canvas.getBoundingClientRect();
  el.edges.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
  el.edges.innerHTML = "";
  const ns = "http://www.w3.org/2000/svg";
  for (const edge of state.graph.edges) {
    const start = socketPoint(edge.source_node, edge.source_port, "output");
    const end = socketPoint(edge.target_node, edge.target_port, "input");
    if (!start || !end) continue;
    const path = document.createElementNS(ns, "path");
    path.setAttribute("class", "edge");
    path.setAttribute("d", cablePath(start, end));
    path.setAttribute("stroke", edgeColor(edge.source_node, edge.source_port));
    el.edges.appendChild(path);
  }
  if (state.wire) {
    const anchor = socketPoint(state.wire.node, state.wire.port, state.wire.kind);
    if (anchor) {
      const cursor = { x: state.wire.x, y: state.wire.y };
      const [start, end] = state.wire.kind === "output" ? [anchor, cursor] : [cursor, anchor];
      const temp = document.createElementNS(ns, "path");
      temp.setAttribute("class", "edge edge-temp");
      temp.setAttribute("d", cablePath(start, end));
      el.edges.appendChild(temp);
    }
  }
}

function socketPoint(nodeId, port, kind) {
  const socket = el.nodes.querySelector(`.socket[data-node="${CSS.escape(nodeId)}"][data-port="${CSS.escape(port)}"][data-kind="${kind}"]`);
  if (!socket) return null;
  const canvas = el.canvas.getBoundingClientRect();
  const rect = socket.getBoundingClientRect();
  return { x: rect.left - canvas.left + rect.width / 2, y: rect.top - canvas.top + rect.height / 2 };
}

function cursorPoint(event) {
  const box = el.canvas.getBoundingClientRect();
  return { x: event.clientX - box.left, y: event.clientY - box.top };
}
