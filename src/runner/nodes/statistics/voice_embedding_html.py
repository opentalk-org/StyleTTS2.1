from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlotPoint:
    x: float
    y: float
    voice: str
    name: str


def render_voice_plot_html(points: list[PlotPoint], title: str, point_size: int) -> bytes:
    voices = sorted({point.voice for point in points})
    colors = _voice_colors(voices)
    payload = {
        "title": title,
        "radius": max(2.0, point_size / 10.0),
        "points": [{"x": point.x, "y": point.y, "v": point.voice, "n": point.name} for point in points],
        "voices": [{"name": voice, "color": colors[voice], "count": sum(1 for p in points if p.voice == voice)} for voice in voices],
    }
    document = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    return document.encode("utf-8")


def _voice_colors(voices: list[str]) -> dict[str, str]:
    total = max(len(voices), 1)
    colors: dict[str, str] = {}
    for index, voice in enumerate(voices):
        hue = (index / total) % 1.0
        saturation = 0.62 if index % 2 == 0 else 0.82
        lightness = 0.58 if index % 3 else 0.48
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[voice] = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
    return colors


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Voice style embeddings</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #ffffff; color: #0f172a; }
  #app { display: grid; grid-template-columns: 1fr 240px; grid-template-rows: auto 1fr; height: 100vh; }
  header { grid-column: 1 / -1; padding: 14px 20px; border-bottom: 1px solid #e2e8f0; }
  header h1 { margin: 0; font-size: 15px; font-weight: 700; }
  header .meta { margin-top: 3px; font-size: 12px; color: #64748b; font-variant-numeric: tabular-nums; }
  #plot { position: relative; overflow: hidden; }
  svg { width: 100%; height: 100%; display: block; touch-action: none; }
  .dot { cursor: pointer; transition: opacity .12s; }
  #legend { border-left: 1px solid #e2e8f0; overflow-y: auto; padding: 10px 6px 10px 12px; }
  #legend .item { display: flex; align-items: center; gap: 7px; padding: 3px 4px; border-radius: 6px;
    font-size: 11.5px; cursor: pointer; }
  #legend .item:hover { background: rgba(100,116,139,.12); }
  #legend .item.off { opacity: .32; }
  #legend .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
  #legend .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, monospace; }
  #legend .ct { margin-left: auto; color: #94a3b8; font-variant-numeric: tabular-nums; }
  #tooltip { position: absolute; pointer-events: none; background: #0f172a; color: #f8fafc; padding: 6px 9px;
    border-radius: 7px; font-size: 11.5px; line-height: 1.35; opacity: 0; transition: opacity .1s;
    box-shadow: 0 6px 18px rgba(0,0,0,.25); max-width: 260px; z-index: 5; }
  #tooltip .v { font-family: ui-monospace, monospace; font-weight: 700; }
  #tooltip .n { color: #cbd5e1; word-break: break-all; }
  .axis { stroke: #cbd5e1; stroke-width: 1; }
  .axis-label { fill: #94a3b8; font-size: 11px; }
  .hint { position: absolute; left: 12px; bottom: 10px; font-size: 11px; color: #94a3b8; }
  @media (prefers-color-scheme: dark) {
    body { background: #0b1120; color: #e2e8f0; }
    header, #legend { border-color: #1e293b; }
    header .meta, #legend .ct, .axis-label, .hint { color: #64748b; }
    .axis { stroke: #1e293b; }
    #tooltip { background: #e2e8f0; color: #0f172a; }
    #tooltip .n { color: #475569; }
  }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1 id="title"></h1>
    <div class="meta" id="meta"></div>
  </header>
  <div id="plot">
    <svg id="svg" preserveAspectRatio="xMidYMid meet"></svg>
    <div id="tooltip"></div>
    <div class="hint">scroll to zoom · drag to pan · click a voice to isolate</div>
  </div>
  <div id="legend"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const SVGNS = "http://www.w3.org/2000/svg";
const W = 1000, H = 700, M = 44;
const svg = document.getElementById("svg");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
document.getElementById("title").textContent = DATA.title;
document.getElementById("meta").textContent =
  `${DATA.points.length} clips · ${DATA.voices.length} voices · PCA of StyleTTS style embeddings`;

const colorOf = {};
DATA.voices.forEach(v => { colorOf[v.name] = v.color; });
const xs = DATA.points.map(p => p.x), ys = DATA.points.map(p => p.y);
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);
const sx = v => M + (xMax === xMin ? 0.5 : (v - xMin) / (xMax - xMin)) * (W - 2 * M);
const sy = v => H - M - (yMax === yMin ? 0.5 : (v - yMin) / (yMax - yMin)) * (H - 2 * M);

const gAxis = document.createElementNS(SVGNS, "g");
[[M, H - M, W - M, H - M], [M, M, M, H - M]].forEach(([x1, y1, x2, y2]) => {
  const l = document.createElementNS(SVGNS, "line");
  l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
  l.setAttribute("class", "axis"); gAxis.appendChild(l);
});
const lx = document.createElementNS(SVGNS, "text");
lx.setAttribute("x", W - M); lx.setAttribute("y", H - M + 26); lx.setAttribute("text-anchor", "end");
lx.setAttribute("class", "axis-label"); lx.textContent = "PC 1"; gAxis.appendChild(lx);
const ly = document.createElementNS(SVGNS, "text");
ly.setAttribute("x", M - 30); ly.setAttribute("y", M - 12); ly.setAttribute("class", "axis-label");
ly.textContent = "PC 2"; gAxis.appendChild(ly);
svg.appendChild(gAxis);

const gPts = document.createElementNS(SVGNS, "g");
svg.appendChild(gPts);
const tooltip = document.getElementById("tooltip");
const plot = document.getElementById("plot");
let hidden = new Set();

const dots = DATA.points.map(p => {
  const c = document.createElementNS(SVGNS, "circle");
  c.setAttribute("cx", sx(p.x)); c.setAttribute("cy", sy(p.y)); c.setAttribute("r", DATA.radius);
  c.setAttribute("fill", colorOf[p.v] || "#888"); c.setAttribute("stroke", "#fff");
  c.setAttribute("stroke-width", 0.6); c.setAttribute("class", "dot"); c.dataset.voice = p.v;
  c.addEventListener("mousemove", ev => {
    tooltip.innerHTML = `<div class="v">${escapeHtml(p.v)}</div><div class="n">${escapeHtml(p.n)}</div>`;
    const r = plot.getBoundingClientRect();
    tooltip.style.left = (ev.clientX - r.left + 12) + "px";
    tooltip.style.top = (ev.clientY - r.top + 12) + "px";
    tooltip.style.opacity = 1;
    c.setAttribute("r", DATA.radius * 1.9);
  });
  c.addEventListener("mouseleave", () => { tooltip.style.opacity = 0; c.setAttribute("r", DATA.radius); });
  gPts.appendChild(c);
  return c;
});

const legend = document.getElementById("legend");
DATA.voices.forEach(v => {
  const item = document.createElement("div");
  item.className = "item";
  item.innerHTML = `<span class="swatch" style="background:${v.color}"></span>` +
    `<span class="nm">${escapeHtml(v.name)}</span><span class="ct">${v.count}</span>`;
  item.addEventListener("click", () => {
    if (hidden.size === 0) { DATA.voices.forEach(o => { if (o.name !== v.name) hidden.add(o.name); }); }
    else if (hidden.has(v.name)) { hidden.delete(v.name); }
    else { hidden.add(v.name); }
    if (hidden.size >= DATA.voices.length) hidden.clear();
    applyFilter();
  });
  item._voice = v.name; legend.appendChild(item);
});
function applyFilter() {
  dots.forEach(d => { d.style.opacity = hidden.has(d.dataset.voice) ? 0.04 : 1; });
  legend.querySelectorAll(".item").forEach(it => it.classList.toggle("off", hidden.has(it._voice)));
}

let view = { x: 0, y: 0, k: 1 };
function applyView() { gPts.setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.k})`); }
svg.addEventListener("wheel", ev => {
  ev.preventDefault();
  const pt = svgPoint(ev);
  const factor = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
  const k = Math.min(20, Math.max(0.5, view.k * factor));
  view.x = pt.x - (pt.x - view.x) * (k / view.k);
  view.y = pt.y - (pt.y - view.y) * (k / view.k);
  view.k = k; applyView();
}, { passive: false });
let drag = null;
svg.addEventListener("pointerdown", ev => { drag = { x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y }; svg.setPointerCapture(ev.pointerId); });
svg.addEventListener("pointermove", ev => {
  if (!drag) return;
  const scale = W / svg.getBoundingClientRect().width;
  view.x = drag.vx + (ev.clientX - drag.x) * scale;
  view.y = drag.vy + (ev.clientY - drag.y) * scale; applyView();
});
svg.addEventListener("pointerup", () => { drag = null; });
function svgPoint(ev) {
  const r = svg.getBoundingClientRect();
  return { x: (ev.clientX - r.left) / r.width * W, y: (ev.clientY - r.top) / r.height * H };
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
</script>
</body>
</html>
"""
