const el = {
  api: document.querySelector("#apiState"),
  palette: document.querySelector("#palette"),
  legend: document.querySelector("#legend"),
  canvas: document.querySelector("#canvas"),
  canvasEmpty: document.querySelector("#canvasEmpty"),
  edges: document.querySelector("#edges"),
  nodes: document.querySelector("#nodes"),
  popup: document.querySelector("#inspectorPopup"),
  popupClose: document.querySelector("#popupCloseBtn"),
  context: document.querySelector("#contextBtn"),
  settings: document.querySelector("#settings"),
  runtimeSettings: document.querySelector("#runtimeSettings"),
  contextSettings: document.querySelector("#contextSettings"),
  tabs: document.querySelectorAll("[data-tab]"),
  tabPanels: document.querySelectorAll("[data-panel]"),
  hint: document.querySelector("#connectHint"),
  runId: document.querySelector("#runId"),
  workDir: document.querySelector("#workDir"),
  outputDir: document.querySelector("#outputDir"),
  runs: document.querySelector("#runs"),
  active: document.querySelector("#activeCount"),
  log: document.querySelector("#log"),
  run: document.querySelector("#runBtn"),
  refresh: document.querySelector("#refreshBtn"),
  clear: document.querySelector("#clearBtn"),
  template: document.querySelector("#templateBtn"),
};

const state = {
  schema: null,
  graph: { nodes: [], edges: [] },
  selection: new Set(),
  wire: null,
  pan: { x: 0, y: 0 },
  seq: 1,
  activeRunId: null,
  runs: [],
  events: [],
  eventAfter: 0,
  runSnapshot: null,
  backendSocket: null,
  runtimeConfig: {},
  rightTab: "settings",
  inspectorOpen: false,
};

function schemaType(prop) {
  if (prop.type) return prop.type;
  const variant = (prop.anyOf || []).find((item) => item.type !== "null");
  return variant ? variant.type : "string";
}

// Signal color a module emits, used to tint its header bar. Falls back to muted
// for sink nodes (no outputs) so the accent reads as "this produces nothing".
function nodeAccent(info) {
  const outputs = Object.values(info.outputs);
  return outputs.length ? typeColor(outputs[0].type) : "#3a4353";
}
