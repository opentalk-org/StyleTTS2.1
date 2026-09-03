import { useEffect, useMemo, useRef, useState } from "react";

import { Icon } from "@/shared/icons";
import { addNode, nodeAccent } from "../graph";
import { useWorkflowStore } from "../store";
import type { WorkflowNodeSchema, WorkflowSchema } from "../types";

type NodeCategoryGroup = {
  category: string;
  nodes: WorkflowNodeSchema[];
};

type PickerPosition = {
  left: number;
  top: number;
  graphX: number;
  graphY: number;
};

const CATEGORY_ORDER = ["Inputs", "Audio", "Text", "ASR", "Synthesis", "Training", "Assets", "Dataset", "Testing"];

function groupedNodes(schema: WorkflowSchema, query: string): NodeCategoryGroup[] {
  const normalized = query.trim().toLowerCase();
  const grouped: Record<string, WorkflowNodeSchema[]> = {};
  Object.values(schema.nodes).forEach((node) => {
    const haystack = `${node.type} ${node.category} ${node.description}`.toLowerCase();
    if (normalized && !haystack.includes(normalized)) return;
    const nodes = grouped[node.category] ?? [];
    nodes.push(node);
    grouped[node.category] = nodes;
  });
  return Object.entries(grouped)
    .sort(([left], [right]) => {
      const leftIndex = CATEGORY_ORDER.indexOf(left);
      const rightIndex = CATEGORY_ORDER.indexOf(right);
      if (leftIndex !== -1 || rightIndex !== -1) {
        const fallbackIndex = CATEGORY_ORDER.length;
        return (leftIndex === -1 ? fallbackIndex : leftIndex) - (rightIndex === -1 ? fallbackIndex : rightIndex);
      }
      return left.localeCompare(right);
    })
    .map(([category, nodes]) => ({
      category,
      nodes: nodes.sort((left, right) => left.type.localeCompare(right.type)),
    }));
}

export function NodePickerPopover({ onClose, position }: { onClose: () => void; position?: PickerPosition }) {
  const { schema } = useWorkflowStore();
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    searchRef.current?.focus();
    const onKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [onClose]);

  useEffect(() => {
    const panel = panelRef.current;
    if (!panel) return;
    const stopCanvasWheel = (event: WheelEvent) => {
      event.stopPropagation();
    };
    panel.addEventListener("wheel", stopCanvasWheel, { passive: true });
    return () => panel.removeEventListener("wheel", stopCanvasWheel);
  }, []);

  const groups = useMemo(() => (schema ? groupedNodes(schema, query) : []), [schema, query]);

  if (!schema) return null;

  const style = position
    ? {
        left: `clamp(12px, ${position.left}px, calc(100vw - 396px))`,
        top: `clamp(12px, ${position.top}px, calc(100vh - 512px))`,
      }
    : {
        left: "50%",
        top: "50%",
        transform: "translate(-50%, -50%)",
      };

  return (
    <div
      className="fixed inset-0 z-[300]"
      role="dialog"
      aria-modal="true"
      aria-label="Add Node"
      onClick={onClose}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <section
        ref={panelRef}
        className="fixed grid max-h-[min(512px,calc(100vh-24px))] w-[min(384px,calc(100vw-24px))] grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-md border border-line bg-panel shadow-[0_18px_58px_rgba(17,24,39,0.22)]"
        style={style}
        onClick={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
        onWheel={(event) => event.stopPropagation()}
      >
        <header className="border-b border-line p-2.5">
          <div className="flex items-center gap-2">
            <label className="relative min-w-0 flex-1">
              <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-txt-mute">
                <Icon name="search" size={15} />
              </span>
              <input
                ref={searchRef}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search Nodes..."
                className="h-8 w-full rounded-md border border-line bg-panel-2 pl-8 pr-2 text-[13px] font-medium text-txt outline-none focus:border-blue-400 focus:bg-panel"
              />
            </label>
            <button
              type="button"
              className="flex h-8 w-8 flex-none cursor-pointer items-center justify-center rounded-md text-txt-mute hover:bg-panel-2 hover:text-txt"
              onClick={onClose}
              aria-label="Close node picker"
            >
              <Icon name="x" size={16} strokeWidth={2.4} />
            </button>
          </div>
        </header>
        <div className="min-h-0 overflow-y-auto p-1.5">
          {groups.length > 0 ? (
            groups.map((group) => (
              <NodeCategory
                key={group.category}
                group={group}
                schema={schema}
                onClose={onClose}
                graphPosition={position}
              />
            ))
          ) : (
            <div className="px-3 py-8 text-center text-[13px] font-medium text-txt-mute">No matching nodes</div>
          )}
        </div>
      </section>
    </div>
  );
}

function NodeCategory({
  group,
  schema,
  onClose,
  graphPosition,
}: {
  group: NodeCategoryGroup;
  schema: WorkflowSchema;
  onClose: () => void;
  graphPosition?: PickerPosition;
}) {
  const { graph, setGraph } = useWorkflowStore();
  return (
    <section className="py-1">
      <header className="sticky top-0 z-10 flex items-center justify-between gap-3 bg-panel/95 px-2 py-1 backdrop-blur">
        <span className="truncate text-[10px] font-bold uppercase text-txt-mute">{group.category}</span>
        <span className="font-mono text-[10px] text-txt-mute">{group.nodes.length}</span>
      </header>
      <div className="grid gap-0.5">
        {group.nodes.map((node) => (
          <button
            key={node.type}
            type="button"
            title={node.description}
            onClick={() => {
              const x = graphPosition?.graphX ?? 80 + graph.nodes.length * 24;
              const y = graphPosition?.graphY ?? 120 + graph.nodes.length * 12;
              setGraph(addNode(schema, graph, node.type, x, y));
              onClose();
            }}
            className="grid h-8 w-full cursor-pointer grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded px-2 text-left hover:bg-panel-2"
          >
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: nodeAccent(schema, node.type) }} />
            <span className="truncate text-[13px] font-semibold text-txt">{node.type}</span>
            <Icon name="plus" size={13} className="text-txt-mute" />
          </button>
        ))}
      </div>
    </section>
  );
}
