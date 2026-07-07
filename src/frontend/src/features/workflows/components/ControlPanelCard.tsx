import { useState } from "react";

import { SchemaForm } from "@/shared/schema-form/SchemaForm";
import type { JsonSchema, SchemaValues } from "@/shared/schema-form/types";
import { Icon } from "@/shared/icons";
import { Input } from "@/shared/ui/Input";
import { Textarea } from "@/shared/ui/Textarea";
import { useWorkflowStore } from "../store";
import type { ControlPanel, WorkflowControl } from "../types";
import { buildWorkflowFieldOverrides } from "./WorkflowFieldPickers";

export function ControlPanelCard({ panel }: { panel: ControlPanel }) {
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const { movePanel, deletePanel, renamePanel } = useWorkflowStore();

  return (
    <article
      onPointerDown={(event) => {
        event.stopPropagation();
        if (!(event.target as HTMLElement).closest("[data-panel-title]")) return;
        setDrag({ x: event.clientX, y: event.clientY });
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!drag) return;
        const zoom = useWorkflowStore.getState().viewport.zoom;
        movePanel(panel.id, (event.clientX - drag.x) / zoom, (event.clientY - drag.y) / zoom);
        setDrag({ x: event.clientX, y: event.clientY });
      }}
      onPointerUp={() => setDrag(null)}
      className="pointer-events-auto absolute w-[300px] select-none rounded-[10px] border-2 border-violet-400/70 bg-panel text-left shadow-[0_14px_40px_rgba(17,24,39,0.16)]"
      style={{ left: panel.x, top: panel.y }}
    >
      <div data-panel-title className="flex cursor-grab items-center gap-2 rounded-t-[8px] border-b border-line bg-violet-50 px-2.5 py-2 active:cursor-grabbing">
        <Icon name="sliders" size={13} strokeWidth={2.2} className="flex-none text-violet-600" />
        <input
          value={panel.title}
          onChange={(event) => renamePanel(panel.id, event.target.value)}
          onPointerDown={(event) => event.stopPropagation()}
          className="min-w-0 flex-1 bg-transparent text-[12.5px] font-bold text-txt outline-none"
        />
        <button
          onPointerDown={(event) => event.stopPropagation()}
          onClick={() => deletePanel(panel.id)}
          title="Delete panel"
          className="flex-none text-txt-mute hover:text-red-500"
        >
          <Icon name="x" size={14} strokeWidth={2.4} />
        </button>
      </div>
      <div className="grid gap-3 p-3">
        {panel.controls.length ? (
          panel.controls.map((control) => <ControlRow key={control.id} panelId={panel.id} control={control} />)
        ) : (
          <p className="text-[11.5px] leading-relaxed text-txt-mute">
            Empty. In a node's Settings, use <span className="font-semibold">"Send to panel"</span> to wire a setting here.
          </p>
        )}
      </div>
    </article>
  );
}

function ControlRow({ panelId, control }: { panelId: string; control: WorkflowControl }) {
  const { schema, graph, patchNode, removeControl, updateControl } = useWorkflowStore();
  const target = control.targets[0];
  const node = graph.nodes.find((item) => item.id === target?.node_id);
  if (!schema || !node || !target) return null;
  const info = schema.nodes[node.type];
  const prop = info?.settings.properties?.[target.key] as JsonSchema | undefined;
  if (!info || !prop) return null;

  const label = control.label.trim() || String(prop.title ?? target.key);
  const description = control.description?.trim() ?? "";
  const fieldSchema: JsonSchema = {
    ...info.settings,
    properties: {
      [target.key]: {
        ...prop,
        title: label,
        description: description || prop.description,
      },
    },
  };
  const write = (value: unknown) => {
    for (const item of control.targets) {
      const bound = useWorkflowStore.getState().graph.nodes.find((entry) => entry.id === item.node_id);
      if (!bound) continue;
      patchNode(item.node_id, { params: { ...bound.params, [item.key]: value } as SchemaValues });
    }
  };

  return (
    <div className="rounded-md border border-line bg-panel-2/40 p-2">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="min-w-0 truncate font-mono text-[10px] text-txt-mute" title={control.targets.map((item) => `${item.node_id}.${item.key}`).join(", ")}>
          {control.targets.length > 1 ? `${control.targets.length}× ` : ""}
          {control.targets.map((item) => item.node_id).join(", ")}
        </span>
        <button onClick={() => removeControl(panelId, control.id)} title="Unwire" className="flex-none text-txt-mute hover:text-red-500">
          <Icon name="x" size={12} strokeWidth={2.4} />
        </button>
      </div>
      <div className="mb-2 grid gap-1.5">
        <Input
          filled
          value={control.label}
          placeholder={String(prop.title ?? target.key)}
          onChange={(event) => updateControl(panelId, control.id, { label: event.target.value })}
          onPointerDown={(event) => event.stopPropagation()}
          className="h-8 px-2 font-semibold"
        />
        <Textarea
          value={control.description ?? ""}
          placeholder="Description"
          onChange={(event) => updateControl(panelId, control.id, { description: event.target.value })}
          onPointerDown={(event) => event.stopPropagation()}
          className="min-h-[46px] resize-none px-2 py-1.5 text-[12px] leading-snug"
        />
      </div>
      <SchemaForm
        schema={fieldSchema}
        values={{ [target.key]: node.params[target.key] }}
        onChange={(values) => write(values[target.key])}
        overrides={buildWorkflowFieldOverrides(info.settings)}
      />
    </div>
  );
}
