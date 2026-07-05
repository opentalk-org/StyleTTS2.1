import { useEffect } from "react";

import { Icon } from "@/shared/icons";
import { addNode, nodeAccent } from "../logic";
import { useWorkflowStore } from "../store";
import type { WorkflowNodeSchema, WorkflowSchema } from "../types";

type CategoryTheme = {
  tint: string;
  border: string;
  text: string;
};

type NodeCategoryGroup = {
  category: string;
  nodes: WorkflowNodeSchema[];
  theme: CategoryTheme;
};

const CATEGORY_THEMES: CategoryTheme[] = [
  { tint: "#eff6ff", border: "#3b82f6", text: "#1d4ed8" },
  { tint: "#ecfdf5", border: "#10b981", text: "#047857" },
  { tint: "#fffbeb", border: "#f59e0b", text: "#b45309" },
  { tint: "#fef2f2", border: "#ef4444", text: "#b91c1c" },
  { tint: "#f5f3ff", border: "#8b5cf6", text: "#6d28d9" },
  { tint: "#ecfeff", border: "#06b6d4", text: "#0e7490" },
];

function groupedNodes(schema: WorkflowSchema): NodeCategoryGroup[] {
  const grouped: Record<string, WorkflowNodeSchema[]> = {};
  Object.values(schema.nodes).forEach((node) => {
    const nodes = grouped[node.category] ?? [];
    nodes.push(node);
    grouped[node.category] = nodes;
  });
  return Object.entries(grouped)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([category, nodes], index) => {
      const theme = CATEGORY_THEMES[index % CATEGORY_THEMES.length];
      if (!theme) throw new Error(`Missing theme for category: ${category}`);
      return {
        category,
        theme,
        nodes: nodes.sort((left, right) => left.type.localeCompare(right.type)),
      };
    });
}

export function NodePickerPopover({ onClose }: { onClose: () => void }) {
  const { schema } = useWorkflowStore();
  useEffect(() => {
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [onClose]);
  if (!schema) return null;
  const groups = groupedNodes(schema);
  return (
    <div
      className="fixed inset-0 z-[300] flex items-center justify-center bg-gray-900/45 p-6"
      role="dialog"
      aria-modal="true"
      aria-label="Add workflow node"
      onClick={onClose}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        className="grid max-h-[86vh] w-[min(920px,calc(100vw-32px))] grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-lg border border-line bg-panel shadow-[0_26px_90px_rgba(17,24,39,0.26)]"
        onClick={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-50 text-blue-600">
                <Icon name="plus" size={17} strokeWidth={2.4} />
              </span>
              <h2 className="text-[18px] font-bold text-txt">Add node</h2>
            </div>
          </div>
          <button
            type="button"
            className="flex h-8 w-8 flex-none cursor-pointer items-center justify-center rounded-md text-txt-mute hover:bg-panel-2 hover:text-txt"
            onClick={onClose}
            aria-label="Close node picker"
          >
            <Icon name="x" size={17} strokeWidth={2.4} />
          </button>
        </header>
        <div className="min-h-0 overflow-y-auto p-5">
          <div className="grid gap-4 md:grid-cols-2">
            {groups.map((group) => (
              <NodeCategory key={group.category} group={group} schema={schema} onClose={onClose} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function NodeCategory({ group, schema, onClose }: { group: NodeCategoryGroup; schema: WorkflowSchema; onClose: () => void }) {
  const { graph, setGraph } = useWorkflowStore();
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-panel">
      <header className="flex items-center justify-between gap-3 border-b border-line px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="h-8 w-1.5 rounded-full" style={{ backgroundColor: group.theme.border }} />
          <span
            className="truncate rounded-md px-2 py-1 text-[12px] font-bold uppercase"
            style={{ backgroundColor: group.theme.tint, color: group.theme.text }}
          >
            {group.category}
          </span>
        </div>
        <span className="font-mono text-[11px] text-txt-mute">{group.nodes.length}</span>
      </header>
      <div className="grid gap-1 p-2">
        {group.nodes.map((node) => (
          <button
            key={node.type}
            type="button"
            onClick={() => {
              setGraph(addNode(schema, graph, node.type, 80 + graph.nodes.length * 24, 120 + graph.nodes.length * 12));
              onClose();
            }}
            className="grid min-h-12 w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-2.5 py-2 text-left hover:bg-panel-2"
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: nodeAccent(schema, node.type) }} />
            <span className="min-w-0">
              <span className="block truncate text-[13px] font-semibold text-txt">{node.type}</span>
              <span className="block truncate text-[11px] text-txt-mute">
                {Object.keys(node.inputs).length} inputs / {Object.keys(node.outputs).length} outputs
              </span>
            </span>
            <Icon name="plus" size={15} className="text-txt-mute" />
          </button>
        ))}
      </div>
    </section>
  );
}
