function dragLoop(onMove) {
  function move(event) {
    onMove(event);
  }
  function up(event) {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    if (dragLoop.onUp) dragLoop.onUp(event);
    dragLoop.onUp = null;
  }
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

function startPan(event) {
  event.preventDefault();
  el.canvas.classList.add("panning");
  const origin = { x: event.clientX, y: event.clientY, px: state.pan.x, py: state.pan.y };
  let moved = false;
  dragLoop.onUp = () => {
    el.canvas.classList.remove("panning");
    if (!moved && state.selection.size) {
      state.selection = new Set();
      render();
    }
  };
  dragLoop((moveEvent) => {
    moved = true;
    state.pan.x = origin.px + moveEvent.clientX - origin.x;
    state.pan.y = origin.py + moveEvent.clientY - origin.y;
    applyPan();
    renderEdges();
  });
}

function startNodeDrag(event) {
  event.preventDefault();
  const start = { x: event.clientX, y: event.clientY };
  const group = state.graph.nodes
    .filter((node) => state.selection.has(node.id))
    .map((node) => ({ node, nx: node.x, ny: node.y, card: el.nodes.querySelector(`.node[data-id="${CSS.escape(node.id)}"]`) }));
  dragLoop.onUp = null;
  dragLoop((moveEvent) => {
    const dx = moveEvent.clientX - start.x;
    const dy = moveEvent.clientY - start.y;
    for (const item of group) {
      item.node.x = item.nx + dx;
      item.node.y = item.ny + dy;
      if (item.card) item.card.style.transform = `translate(${item.node.x}px, ${item.node.y}px)`;
    }
    renderEdges();
  });
}

function nodesInRect(rect) {
  const canvas = el.canvas.getBoundingClientRect();
  const ids = [];
  for (const card of el.nodes.querySelectorAll(".node")) {
    const r = card.getBoundingClientRect();
    const x = r.left - canvas.left;
    const y = r.top - canvas.top;
    if (x < rect.x + rect.w && x + r.width > rect.x && y < rect.y + rect.h && y + r.height > rect.y) ids.push(card.dataset.id);
  }
  return ids;
}

function startMarquee(event) {
  event.preventDefault();
  const origin = cursorPoint(event);
  const box = document.createElement("div");
  box.className = "marquee";
  el.canvas.appendChild(box);
  let rect = { x: origin.x, y: origin.y, w: 0, h: 0 };
  dragLoop.onUp = () => {
    box.remove();
    setSelection(nodesInRect(rect));
    render();
  };
  dragLoop((moveEvent) => {
    const p = cursorPoint(moveEvent);
    rect = { x: Math.min(origin.x, p.x), y: Math.min(origin.y, p.y), w: Math.abs(p.x - origin.x), h: Math.abs(p.y - origin.y) };
    box.style.left = `${rect.x}px`;
    box.style.top = `${rect.y}px`;
    box.style.width = `${rect.w}px`;
    box.style.height = `${rect.h}px`;
  });
}

function startWire(event, socket) {
  event.preventDefault();
  const data = socket.dataset;
  if (data.kind === "input") {
    const index = state.graph.edges.findIndex((edge) => edge.target_node === data.node && edge.target_port === data.port);
    if (index >= 0) {
      const edge = state.graph.edges.splice(index, 1)[0];
      state.wire = { node: edge.source_node, port: edge.source_port, kind: "output" };
    } else {
      state.wire = { node: data.node, port: data.port, kind: "input" };
    }
  } else {
    state.wire = { node: data.node, port: data.port, kind: "output" };
  }
  const point = cursorPoint(event);
  state.wire.x = point.x;
  state.wire.y = point.y;
  render();

  dragLoop.onUp = (upEvent) => {
    const wire = state.wire;
    state.wire = null;
    const target = document.elementFromPoint(upEvent.clientX, upEvent.clientY);
    if (wire && target && target.classList.contains("socket")) {
      const t = target.dataset;
      if (wire.kind === "output" && t.kind === "input") return connect(wire.node, wire.port, t.node, t.port);
      if (wire.kind === "input" && t.kind === "output") return connect(t.node, t.port, wire.node, wire.port);
    }
    render();
  };
  dragLoop((moveEvent) => {
    if (!state.wire) return;
    const q = cursorPoint(moveEvent);
    state.wire.x = q.x;
    state.wire.y = q.y;
    renderEdges();
  });
}

function onNodesPointerDown(event) {
  const socket = event.target.closest(".socket");
  if (socket) {
    event.stopPropagation();
    return startWire(event, socket);
  }
  const card = event.target.closest(".node");
  if (!card || event.target.closest(".node-del")) return;
  const id = card.dataset.id;
  event.stopPropagation();

  if (event.shiftKey || event.metaKey || event.ctrlKey) {
    toggleSelection(id);
    render();
    return;
  }
  if (!state.selection.has(id)) setSelection([id]);
  render();

  if (event.target.closest(".node-title")) startNodeDrag(event);
}

function onNodesClick(event) {
  const del = event.target.closest(".node-del");
  if (del) {
    event.stopPropagation();
    deleteNode(del.closest(".node").dataset.id);
  }
}

function onCanvasPointerDown(event) {
  if (event.target !== el.canvas && event.target !== el.canvasEmpty) return;
  if (event.shiftKey) startMarquee(event);
  else startPan(event);
}
