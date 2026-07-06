import { useLayoutEffect, useRef, useState } from "react";

import { nodeSnapshot } from "../api";
import { graphPoint } from "../logic";
import { useLoadNodeMutation, useUnloadNodeMutation } from "../query";
import { useWorkflowStore } from "../store";
import type { PortAnchorKey, WorkflowNode } from "../types";

type NodeStatusTone = "idle" | "running" | "stopped" | "failed";

export function WorkflowNodeCard({ node }: { node: WorkflowNode }) {
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const loadNode = useLoadNodeMutation();
  const unloadNode = useUnloadNodeMutation();
  const { schema, selectedNodeIds, activeRunId, snapshots, selectNode, moveSelection, startWire, deleteSelection, openInspector } = useWorkflowStore();
  if (!schema) return null;
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const active = selectedNodeIds.includes(node.id);
  const snapshot = nodeSnapshot(activeRunId ? snapshots[activeRunId] : undefined, node.id);
  const inputs = Object.values(info.inputs);
  const outputs = Object.values(info.outputs);
  const colorFor = (type: string) => {
    const schemaType = schema.types[type];
    if (!schemaType) throw new Error(`Unknown port type: ${type}`);
    return schemaType.color;
  };

  const completed = snapshot?.counters.tasks_completed ?? snapshot?.counters.packets_created ?? 0;
  const queued = snapshot?.queue_size ?? 0;
  const left = snapshot?.remaining_items ?? "-";
  const loaded = snapshot?.loaded ? "loaded" : "unloaded";
  const status = nodeStatusTone(snapshot?.status);

  return (
    <article
      onClick={(event) => {
        event.stopPropagation();
        selectNode(node.id, event.metaKey || event.ctrlKey || event.shiftKey);
      }}
      onPointerDown={(event) => {
        event.stopPropagation();
        if (!active) selectNode(node.id, event.metaKey || event.ctrlKey || event.shiftKey);
        if (!(event.target as HTMLElement).closest("[data-node-title]")) return;
        setDrag({ x: event.clientX, y: event.clientY });
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!drag) return;
        const zoom = useWorkflowStore.getState().viewport.zoom;
        moveSelection((event.clientX - drag.x) / zoom, (event.clientY - drag.y) / zoom);
        setDrag({ x: event.clientX, y: event.clientY });
      }}
      onPointerUp={() => setDrag(null)}
      className={`group pointer-events-auto absolute w-max min-w-[220px] max-w-[420px] cursor-pointer select-none rounded-[9px] border bg-panel text-left shadow-[0_12px_34px_rgba(17,24,39,0.14)] ${active ? "border-amber-500 ring-1 ring-amber-500" : status === "failed" ? "border-red-500" : status === "stopped" ? "border-amber-400" : "border-line-2"} ${status === "running" ? "shadow-[0_0_0_2px_rgba(79,209,197,0.45),0_16px_40px_rgba(17,24,39,0.18)]" : ""} ${status === "failed" ? "shadow-[0_0_0_2px_rgba(231,111,81,0.9),0_16px_40px_rgba(17,24,39,0.22)]" : ""}`}
      style={{ left: node.x, top: node.y }}
    >
      <div data-node-title className="flex cursor-grab items-center gap-2 border-b border-line px-2.5 py-2 pr-7 active:cursor-grabbing">
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <strong className="min-w-0 truncate font-mono text-[12px] text-txt">{node.id}</strong>
          {status === "failed" ? <span className="flex-none rounded-full bg-red-100 px-1.5 py-0.5 font-mono text-[9px] font-extrabold uppercase text-red-700">failed</span> : null}
          {info.is_input ? <span className="flex-none rounded-full bg-blue-100 px-1.5 py-0.5 font-mono text-[9px] font-extrabold uppercase text-blue-700">input</span> : null}
          <span className="ml-auto min-w-0 truncate font-mono text-[10px] text-txt-mute">{node.type}</span>
        </div>
        <button
          type="button"
          className="absolute right-1.5 top-1.5 cursor-pointer border-0 bg-transparent px-1 py-0.5 text-[16px] leading-none text-txt-mute opacity-0 hover:text-red-500 group-hover:opacity-100"
          aria-label="Delete node"
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            if (!active) selectNode(node.id);
            deleteSelection();
          }}
        >
          &times;
        </button>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(92px,auto)_auto] gap-1.5 border-b border-line bg-panel-2/50 px-2.5 py-2">
        <Metric label={info.is_input ? "left" : "queued"} value={info.is_input ? left : queued} tone={status} role={info.is_input ? "items" : "queue"} />
        <Metric label="done" value={completed} tone={status} role="state" />
        <Lifecycle
          loaded={loaded}
          status={status}
          disabled={!activeRunId}
          onLoad={() => activeRunId && loadNode.mutate({ runId: activeRunId, nodeId: node.id })}
          onUnload={() => activeRunId && unloadNode.mutate({ runId: activeRunId, nodeId: node.id })}
        />
        <button
          type="button"
          className="cursor-pointer rounded-md bg-panel px-2 py-1 font-mono text-[10px] font-extrabold uppercase leading-none text-txt-mute hover:text-amber-600"
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            selectNode(node.id);
            openInspector("settings");
          }}
        >
          edit
        </button>
      </div>
      <div className="grid grid-cols-[minmax(90px,1fr)_minmax(90px,1fr)] gap-x-2 px-1 py-2 font-mono text-[11px] text-txt-dim">
        <div className="grid gap-1 inputs">
          {inputs.map((port) => (
            <button
              key={port.name}
              type="button"
              className="flex min-h-[22px] cursor-crosshair items-center gap-[7px] text-left"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => event.stopPropagation()}
            >
              <Socket nodeId={node.id} portName={port.name} kind="input" nodeX={node.x} nodeY={node.y} color={colorFor(port.type)} onStart={startWire} />
              <span className="truncate">{port.name}</span>
            </button>
          ))}
        </div>
        <div className="grid justify-items-end gap-1 outputs">
          {outputs.map((port) => (
            <button
              key={port.name}
              type="button"
              className="flex min-h-[22px] cursor-crosshair items-center justify-end gap-[7px] text-right"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => event.stopPropagation()}
            >
              <span className="truncate">{port.name}</span>
              <Socket nodeId={node.id} portName={port.name} kind="output" nodeX={node.x} nodeY={node.y} color={colorFor(port.type)} onStart={startWire} />
            </button>
          ))}
        </div>
      </div>
      {status === "failed" && snapshot?.error ? (
        <div className="border-t border-red-200 bg-red-50 px-2.5 py-1.5 font-mono text-[10px] leading-snug text-red-700">
          <span className="line-clamp-2 break-words">{snapshot.error}</span>
        </div>
      ) : null}
    </article>
  );
}

function nodeStatusTone(status: string | undefined): NodeStatusTone {
  if (status === "running" || status === "failed" || status === "stopped") return status;
  return "idle";
}

function Metric({ label, value, tone, role }: { label: string; value: number | string; tone: NodeStatusTone; role: string }) {
  const color = role === "queue" ? "text-emerald-700" : role === "items" ? "text-blue-700" : tone === "running" ? "text-emerald-700" : tone === "failed" ? "text-red-600" : tone === "stopped" ? "text-amber-700" : "text-txt";
  return (
    <div className={`grid min-w-0 place-content-center gap-0.5 rounded-md border bg-panel p-1.5 text-center font-mono ${tone === "running" ? "border-emerald-400" : tone === "failed" ? "border-red-400" : tone === "stopped" ? "border-amber-400" : "border-line"}`}>
      <strong className={`overflow-hidden text-ellipsis whitespace-nowrap text-[13px] leading-none ${color}`}>{value}</strong>
      <span className="text-[9px] uppercase text-txt-mute">{label}</span>
    </div>
  );
}

function Lifecycle({
  loaded,
  status,
  disabled,
  onLoad,
  onUnload,
}: {
  loaded: string;
  status: NodeStatusTone;
  disabled: boolean;
  onLoad: () => void;
  onUnload: () => void;
}) {
  return (
    <div className="grid grid-cols-2 content-start gap-1">
      <span className={`col-span-2 flex min-h-[22px] items-center justify-center rounded-full px-1.5 text-[10px] font-extrabold leading-tight ${loaded === "loaded" ? "bg-blue-100 text-blue-700" : "bg-panel text-txt-mute"}`}>{loaded}</span>
      <span className={`col-span-2 flex min-h-[22px] items-center justify-center rounded-full px-1.5 text-[10px] font-extrabold leading-tight ${status === "running" ? "bg-emerald-100 text-emerald-700" : status === "failed" ? "bg-red-100 text-red-700" : status === "stopped" ? "bg-amber-100 text-amber-700" : "bg-panel text-txt-mute"}`}>{status}</span>
      <LifecycleButton disabled={disabled} onClick={onLoad}>load</LifecycleButton>
      <LifecycleButton disabled={disabled} onClick={onUnload}>unload</LifecycleButton>
    </div>
  );
}

function LifecycleButton({ disabled, onClick, children }: { disabled: boolean; onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      disabled={disabled}
      className="min-w-0 cursor-pointer rounded border border-line bg-panel px-1.5 py-1 font-mono text-[9px] font-extrabold uppercase text-txt-mute hover:border-amber-500 hover:text-amber-600 disabled:cursor-not-allowed disabled:opacity-40"
      onPointerDown={(event) => event.stopPropagation()}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      {children}
    </button>
  );
}

function Socket({
  nodeId,
  portName,
  kind,
  nodeX,
  nodeY,
  color,
  onStart,
}: {
  nodeId: string;
  portName: string;
  kind: "input" | "output";
  nodeX: number;
  nodeY: number;
  color: string;
  onStart: (nodeId: string, portName: string, kind: "input" | "output", x: number, y: number) => void;
}) {
  const ref = useRef<HTMLSpanElement | null>(null);
  const viewport = useWorkflowStore((state) => state.viewport);
  const setPortAnchor = useWorkflowStore((state) => state.setPortAnchor);
  useLayoutEffect(() => {
    const socket = ref.current;
    const canvas = document.querySelector<HTMLElement>("[data-workflow-canvas]");
    if (!socket || !canvas) return;
    const socketRect = socket.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const key = `${nodeId}:${portName}:${kind}` as PortAnchorKey;
    setPortAnchor(key, {
      x: socketRect.left - canvasRect.left + socketRect.width / 2,
      y: socketRect.top - canvasRect.top + socketRect.height / 2,
    });
  }, [kind, nodeId, nodeX, nodeY, portName, setPortAnchor, viewport.x, viewport.y, viewport.zoom]);
  return (
    <span
      ref={ref}
      data-workflow-socket
      data-node={nodeId}
      data-port={portName}
      data-kind={kind}
      className={`h-[14px] w-[14px] flex-none cursor-crosshair rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(17,24,39,0.45),0_1px_3px_rgba(17,24,39,0.25)] transition-transform hover:scale-125 ${kind === "input" ? "-ml-[12px]" : "-mr-[12px]"}`}
      onPointerDown={(event) => {
        event.preventDefault();
        event.stopPropagation();
        const canvas = document.querySelector<HTMLElement>("[data-workflow-canvas]");
        if (!canvas) return;
        const box = canvas.getBoundingClientRect();
        const point = graphPoint(useWorkflowStore.getState().viewport, event.clientX, event.clientY, box.left, box.top);
        onStart(nodeId, portName, kind, point.x, point.y);
      }}
      style={{ backgroundColor: color }}
    />
  );
}
