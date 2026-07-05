import { SchemaForm } from "@/shared/schema-form/SchemaForm";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { nodeSnapshot } from "../api";
import { useLoadNodeMutation, useUnloadNodeMutation } from "../query";
import { useWorkflowStore } from "../store";

export function WorkflowInspector() {
  const loadNode = useLoadNodeMutation();
  const unloadNode = useUnloadNodeMutation();
  const { schema, graph, selectedNodeIds, activeRunId, snapshots, setGraph, deleteSelection, renameSelectedNode } = useWorkflowStore();
  if (!schema || selectedNodeIds.length === 0) return null;
  if (selectedNodeIds.length > 1) {
    return (
      <aside className="w-[360px] flex-none border-l border-line bg-panel p-4">
        <div className="text-[14px] font-bold text-txt">{selectedNodeIds.length} nodes selected</div>
        <Button className="mt-3" variant="secondary" icon="trash" onClick={deleteSelection}>Delete selected</Button>
      </aside>
    );
  }
  const node = graph.nodes.find((item) => item.id === selectedNodeIds[0]);
  if (!node) return null;
  const info = schema.nodes[node.type];
  if (!info) throw new Error(`Unknown node type: ${node.type}`);
  const snapshot = nodeSnapshot(activeRunId ? snapshots[activeRunId] : undefined, node.id);
  const update = (patch: Partial<typeof node>) => setGraph({ ...graph, nodes: graph.nodes.map((item) => (item.id === node.id ? { ...item, ...patch } : item)) });

  return (
    <aside className="w-[360px] flex-none overflow-y-auto border-l border-line bg-panel p-4">
      <div className="mb-3 flex items-center gap-2">
        <div className="flex-1 text-[14px] font-bold text-txt">{node.type}</div>
        <Button variant="secondary" icon="trash" onClick={deleteSelection}>Delete</Button>
      </div>
      <div className="grid gap-4">
        <Input filled className="h-9" value={node.id} onChange={(event) => renameSelectedNode(event.target.value)} />
        <div className="flex gap-2">
          <Button disabled={!activeRunId || snapshot?.loaded === true} onClick={() => activeRunId && loadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Load</Button>
          <Button disabled={!activeRunId || snapshot?.loaded !== true} onClick={() => activeRunId && unloadNode.mutate({ runId: activeRunId, nodeId: node.id })}>Unload</Button>
        </div>
        <div>
          <div className="mb-2 text-[12px] font-bold uppercase tracking-wider text-txt-mute">Settings</div>
          <SchemaForm schema={info.settings} values={node.params} onChange={(params) => update({ params })} />
        </div>
        <div>
          <div className="mb-2 text-[12px] font-bold uppercase tracking-wider text-txt-mute">Node runtime</div>
          <SchemaForm schema={info.runtime} values={node.runtime} onChange={(runtime) => update({ runtime })} />
        </div>
      </div>
    </aside>
  );
}
