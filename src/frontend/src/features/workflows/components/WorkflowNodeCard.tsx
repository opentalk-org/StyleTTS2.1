import { useState } from "react";

import { Icon } from "@/shared/icons";
import { nodeSnapshot } from "../api";
import { nodeAccent } from "../logic";
import { useWorkflowStore } from "../store";
import type { WorkflowNode } from "../types";

export function WorkflowNodeCard({ node }: { node: WorkflowNode }) {
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const { schema, selectedNodeIds, activeRunId, snapshots, wireDraft, selectNode, moveSelection, setWireDraft, addEdge } = useWorkflowStore();
  if (!schema) return null;
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const active = selectedNodeIds.includes(node.id);
  const accent = nodeAccent(schema, node.type);
  const snapshot = nodeSnapshot(activeRunId ? snapshots[activeRunId] : undefined, node.id);
  const inputs = Object.values(info.inputs);
  const outputs = Object.values(info.outputs);
  const colorFor = (type: string) => {
    const schemaType = schema.types[type];
    if (!schemaType) throw new Error(`Unknown port type: ${type}`);
    return schemaType.color;
  };

  return (
    <article
      onClick={(event) => {
        event.stopPropagation();
        selectNode(node.id, event.metaKey || event.ctrlKey || event.shiftKey);
      }}
      onPointerDown={(event) => {
        event.stopPropagation();
        if (!active) selectNode(node.id, event.metaKey || event.ctrlKey || event.shiftKey);
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
      className={`absolute w-[240px] rounded-md border bg-panel text-left shadow-sm ${active ? "border-blue-500" : "border-line"}`}
      style={{ left: node.x, top: node.y }}
    >
      <div className="h-1 rounded-t-md" style={{ backgroundColor: accent }} />
      <div className="border-b border-line px-3 py-2">
        <div className="flex items-center gap-2">
          <Icon name={info.is_input ? "database" : "workflow"} size={15} className="text-txt-mute" />
          <div className="min-w-0">
            <div className="truncate text-[13px] font-bold text-txt">{node.id}</div>
            <div className="truncate text-[11px] text-txt-mute">{node.type}</div>
          </div>
        </div>
      </div>
      {snapshot ? (
        <div className="border-b border-line px-3 py-2 text-[11px] text-txt-dim">
          <span className="font-semibold text-txt">{snapshot.status}</span>
          <span className="ml-2">q {snapshot.queue_size}</span>
          <span className="ml-2">left {snapshot.remaining_items ?? "-"}</span>
          <span className="ml-2">{snapshot.loaded ? "loaded" : "cold"}</span>
        </div>
      ) : null}
      <div className="grid grid-cols-2 gap-4 px-3 py-2 text-[11px] text-txt-dim">
        <div className="grid gap-1">
          {inputs.map((port) => (
            <button
              key={port.name}
              type="button"
              className="flex items-center gap-1 text-left"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                if (wireDraft) addEdge({ source_node: wireDraft.source_node, source_port: wireDraft.source_port, target_node: node.id, target_port: port.name });
              }}
            >
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colorFor(port.type) }} />
              <span className="truncate">{port.name}</span>
            </button>
          ))}
        </div>
        <div className="grid gap-1">
          {outputs.map((port) => (
            <button
              key={port.name}
              type="button"
              className="flex items-center justify-end gap-1 text-right"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                setWireDraft({ source_node: node.id, source_port: port.name, x: node.x + 240, y: node.y + 80 });
              }}
            >
              <span className="truncate">{port.name}</span>
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colorFor(port.type) }} />
            </button>
          ))}
        </div>
      </div>
    </article>
  );
}
