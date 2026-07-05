import { useEffect, useState, type ReactNode } from "react";

import { SchemaForm } from "@/shared/schema-form/SchemaForm";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { fetchNodeLog, nodeSnapshot } from "../api";
import { useLoadNodeMutation, useUnloadNodeMutation } from "../query";
import { useWorkflowStore } from "../store";

export function WorkflowInspector() {
  const loadNode = useLoadNodeMutation();
  const unloadNode = useUnloadNodeMutation();
  const [log, setLog] = useState<{ content: string; truncated: boolean; error: string | null } | null>(null);
  const { schema, graph, selectedNodeIds, activeRunId, snapshots, inspectorOpen, inspectorTab, setInspectorTab, closeInspector, setGraph, deleteSelection, renameSelectedNode } = useWorkflowStore();
  const node = graph.nodes.find((item) => item.id === selectedNodeIds[0]);
  useEffect(() => {
    setLog(null);
    if (!activeRunId || !node || inspectorTab !== "logs") return;
    fetchNodeLog(activeRunId, node.id).then(setLog).catch((error) => setLog({ content: "", truncated: false, error: error.message }));
  }, [activeRunId, inspectorTab, node]);
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
  const update = (patch: Partial<typeof node>) => setGraph({ ...graph, nodes: graph.nodes.map((item) => (item.id === node.id ? { ...item, ...patch } : item)) });

  return (
    <InspectorShell title={`${node.id} · ${node.type}`} onClose={closeInspector}>
      <div className="grid grid-cols-3 gap-1.5">
        <Tab active={inspectorTab === "settings"} onClick={() => setInspectorTab("settings")}>Settings</Tab>
        <Tab active={inspectorTab === "runtime"} onClick={() => setInspectorTab("runtime")}>Runtime</Tab>
        <Tab active={inspectorTab === "logs"} onClick={() => setInspectorTab("logs")}>Logs</Tab>
      </div>
      {inspectorTab === "settings" ? (
        <div className="grid gap-3">
          <label className="grid gap-1.5 font-mono text-[10px] font-bold uppercase tracking-wide text-txt-mute">
            node id
            <Input filled className="h-9" value={node.id} onChange={(event) => renameSelectedNode(event.target.value)} />
          </label>
          <SchemaForm schema={info.settings} values={node.params} onChange={(params) => update({ params })} />
          <Button variant="secondary" icon="trash" onClick={deleteSelection}>Delete</Button>
        </div>
      ) : null}
      {inspectorTab === "runtime" ? (
        <div className="grid gap-3">
          <div className="flex gap-2">
            <Button disabled={!activeRunId || snapshot?.loaded === true} onClick={() => activeRunId && loadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Load</Button>
            <Button disabled={!activeRunId || snapshot?.loaded !== true} onClick={() => activeRunId && unloadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Unload</Button>
          </div>
          <SchemaForm schema={info.runtime} values={node.runtime} onChange={(runtime) => update({ runtime })} />
        </div>
      ) : null}
      {inspectorTab === "logs" ? (
        <div className="grid gap-2">
          {!activeRunId ? <p className="font-mono text-[12px] text-txt-mute">Start or select a run to view node logs.</p> : null}
          {log?.truncated ? <p className="font-mono text-[11px] text-amber-700">Showing latest 1 MB.</p> : null}
          {log?.error ? <p className="font-mono text-[11px] text-red-600">{log.error}</p> : null}
          <pre className="min-h-[340px] max-h-[calc(100vh-220px)] overflow-auto rounded-md border border-line bg-panel-2 p-3 font-mono text-[11px] leading-relaxed text-txt-dim">
            {log?.content || "No log lines for this node yet."}
          </pre>
        </div>
      ) : null}
    </InspectorShell>
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

function Tab({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      className={`min-h-[30px] rounded-md border font-mono text-[10px] font-extrabold uppercase ${active ? "border-amber-500 bg-amber-50 text-amber-700" : "border-line bg-panel-2 text-txt-mute"}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
