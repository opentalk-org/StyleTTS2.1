import { useEffect, useState, type ReactNode } from "react";

import { SchemaForm } from "@/shared/schema-form/SchemaForm";
import type { JsonSchema } from "@/shared/schema-form/types";
import { Button } from "@/shared/ui/Button";
import { Field } from "@/shared/ui/form/Field";
import { FormSection } from "@/shared/ui/FormSection";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { Tabs } from "@/shared/ui/Tabs";
import { copyText } from "@/shared/clipboard";
import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import type { WorkflowNode } from "../types";
import { fetchNodeLog, nodeSnapshot } from "../api";
import { useLoadNodeMutation, useUnloadNodeMutation } from "../query";
import { useWorkflowStore } from "../store";
import { buildWorkflowFieldOverrides } from "./WorkflowFieldPickers";

type InspectorTab = "settings" | "runtime" | "logs";

const INSPECTOR_TABS = [
  { value: "settings", label: "Settings" },
  { value: "runtime", label: "Runtime" },
  { value: "logs", label: "Logs" },
];

export function WorkflowInspector() {
  const loadNode = useLoadNodeMutation();
  const unloadNode = useUnloadNodeMutation();
  const [log, setLog] = useState<{ content: string; truncated: boolean; error: string | null } | null>(null);
  const { schema, graph, selectedNodeIds, activeRunId, runs, snapshots, inspectorOpen, inspectorTab, setInspectorTab, closeInspector, patchNode, deleteSelection, renameSelectedNode } = useWorkflowStore();
  const node = graph.nodes.find((item) => item.id === selectedNodeIds[0]);
  const activeRun = activeRunId ? runs.find((run) => run.run_id === activeRunId) : undefined;
  const lifecycleActive = activeRun?.state === "running";
  const nodeId = node?.id;
  useEffect(() => {
    setLog(null);
    if (!activeRunId || !nodeId || inspectorTab !== "logs") return;
    fetchNodeLog(activeRunId, nodeId).then(setLog).catch((error) => setLog({ content: "", truncated: false, error: error.message }));
  }, [activeRunId, inspectorTab, nodeId]);
  if (!schema || !inspectorOpen) return null;
  if (selectedNodeIds.length > 1) {
    return (
      <InspectorShell title={`${selectedNodeIds.length} nodes selected`} onClose={closeInspector}>
        <p className="font-mono text-[12px] text-txt-mute">Drag to move the group, Delete to remove.</p>
        <Button className="mt-3" variant="secondary" icon="trash" onClick={deleteSelection}>Delete selected</Button>
      </InspectorShell>
    );
  }
  if (!node) {
    return (
      <InspectorShell title="Node inspector" onClose={closeInspector}>
        <p className="font-mono text-[12px] text-txt-mute">Select a node to edit its settings.</p>
      </InspectorShell>
    );
  }
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const snapshot = nodeSnapshot(activeRunId ? snapshots[activeRunId] : undefined, node.id);
  const update = (patch: Partial<typeof node>) => patchNode(node.id, patch);
  const downloadLog = () => {
    if (!log?.content) return;
    const blob = new Blob([log.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${node.id}-${activeRunId ?? "run"}.log`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <InspectorShell title={`${node.id} · ${node.type}`} onClose={closeInspector}>
      <div className="flex items-center justify-between gap-3">
        <Tabs value={inspectorTab} onChange={(tab) => setInspectorTab(tab as InspectorTab)} options={INSPECTOR_TABS} />
        {inspectorTab === "logs" ? (
          <Button variant="secondary" icon="download" disabled={!log?.content} onClick={downloadLog}>Download logs</Button>
        ) : null}
      </div>
      {inspectorTab === "settings" ? (
        <div className="flex min-w-0 flex-col gap-3.5">
          {info.description ? (
            <p className="text-[12px] leading-relaxed text-txt-dim">{info.description}</p>
          ) : null}
          <FormSection title="Node identity" tag="Node">
            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-3.5">
              <Field label="Node id">
                <Input filled value={node.id} onChange={(event) => renameSelectedNode(event.target.value)} />
              </Field>
              <Button variant="secondary" icon="trash" onClick={deleteSelection}>Delete</Button>
            </div>
          </FormSection>
          <FormSection title="Settings" tag={node.type}>
            <SchemaForm schema={info.settings} values={node.params} onChange={(params) => update({ params })} overrides={buildWorkflowFieldOverrides(info.settings)} />
          </FormSection>
          <ExposeToPanel node={node} settings={info.settings} />
        </div>
      ) : null}
      {inspectorTab === "runtime" ? (
        <div className="grid gap-3">
          <div className="flex gap-2">
            <Button disabled={!activeRunId || !lifecycleActive || snapshot?.loaded === true} onClick={() => activeRunId && loadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Load</Button>
            <Button disabled={!activeRunId || !lifecycleActive || snapshot?.loaded !== true} onClick={() => activeRunId && unloadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Unload</Button>
          </div>
          <SchemaForm schema={info.runtime} values={node.runtime} onChange={(runtime) => update({ runtime })} />
        </div>
      ) : null}
      {inspectorTab === "logs" ? (
        <div className="grid gap-2">
          {!activeRunId ? <p className="font-mono text-[12px] text-txt-mute">Start or select a run to view node logs.</p> : null}
          {log?.truncated ? <p className="font-mono text-[11px] text-amber-700">Showing latest 1 MB.</p> : null}
          {log?.error ? <p className="font-mono text-[11px] text-red-600">{log.error}</p> : null}
          <div className="relative">
            <button
              type="button"
              title="Copy logs"
              disabled={!log?.content}
              onClick={async () => {
                if (log?.content && (await copyText(log.content))) showToast("Node logs copied");
              }}
              className="absolute right-2 top-2 z-10 flex h-7 w-7 items-center justify-center rounded-md border border-line bg-panel text-txt-mute shadow-sm transition hover:text-txt disabled:opacity-40"
            >
              <Icon name="copy" size={13} strokeWidth={2.2} />
            </button>
            <pre className="min-h-[340px] max-h-[calc(100vh-220px)] overflow-auto rounded-md border border-line bg-panel-2 p-3 pr-11 font-mono text-[11px] leading-relaxed text-txt-dim">
              {log?.content || "No log lines for this node yet."}
            </pre>
          </div>
        </div>
      ) : null}
    </InspectorShell>
  );
}

function ExposeToPanel({ node, settings }: { node: WorkflowNode; settings: JsonSchema }) {
  const { graph, viewport, exposeSetting } = useWorkflowStore();
  const keys = Object.keys(settings.properties ?? {});
  const panels = graph.panels ?? [];
  const [key, setKey] = useState(keys[0] ?? "");
  const [panelId, setPanelId] = useState("");
  useEffect(() => {
    if (!keys.includes(key)) setKey(keys[0] ?? "");
  }, [node.id]);
  if (!keys.length) return null;
  const send = () => {
    if (!key) return;
    const label = String(settings.properties?.[key]?.title ?? key);
    const position = { x: (360 - viewport.x) / viewport.zoom, y: (160 - viewport.y) / viewport.zoom };
    exposeSetting(panelId || null, { node_id: node.id, key }, label, position);
  };
  return (
    <FormSection title="Expose to control panel" tag="Wire">
      <div className="grid grid-cols-[1fr_1fr_auto] items-end gap-2.5">
        <Field label="Setting">
          <Select value={key} onChange={setKey} options={keys.map((item) => ({ value: item, label: String(settings.properties?.[item]?.title ?? item) }))} />
        </Field>
        <Field label="Panel">
          <Select value={panelId} onChange={setPanelId} options={[{ value: "", label: "＋ New panel" }, ...panels.map((panel) => ({ value: panel.id, label: panel.title }))]} />
        </Field>
        <Button variant="secondary" icon="sliders" onClick={send}>Send</Button>
      </div>
    </FormSection>
  );
}

function InspectorShell({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <div className="fixed inset-0 z-30">
      <button className="absolute inset-0 bg-gray-900/30" aria-label="Close inspector" onClick={onClose} />
      <section className="absolute bottom-8 right-8 top-8 grid w-[min(760px,calc(100vw-64px))] grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-auto rounded-lg border border-line bg-panel p-4 shadow-[0_22px_80px_rgba(17,24,39,0.22)]">
        <div className="flex items-center justify-between gap-3">
          <strong className="text-[13px] text-txt">{title}</strong>
          <Button variant="ghost" icon="x" onClick={onClose}>Close</Button>
        </div>
        <div className="grid content-start gap-3 overflow-auto">{children}</div>
      </section>
    </div>
  );
}
