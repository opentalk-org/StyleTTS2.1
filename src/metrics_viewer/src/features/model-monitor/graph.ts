import { MarkerType, type Edge, type Node } from "@xyflow/react";

import type { ModelComponent } from "@/shared/types";

export function visibleComponents(components: ModelComponent[], expanded: Set<string>, search: string): ModelComponent[] {
  if (search.length > 0) {
    const needle = search.toLowerCase();
    return components.filter(
      (item) => item.id.toLowerCase().includes(needle) || item.module_type.toLowerCase().includes(needle),
    );
  }
  const byId = new Map(components.map((item) => [item.id, item]));
  return components.filter((item) => ancestorsExpanded(item, byId, expanded));
}

function ancestorsExpanded(component: ModelComponent, byId: Map<string, ModelComponent>, expanded: Set<string>): boolean {
  if (component.parent_id === null) return true;
  const parent = byId.get(component.parent_id);
  return parent !== undefined && expanded.has(parent.id) && ancestorsExpanded(parent, byId, expanded);
}

export const NODE_WIDTH = 260;
export const NODE_HEIGHT = 58;

export interface GraphNodeData extends Record<string, unknown> {
  id: string;
  moduleType: string;
  parameterCount: number;
  hasChildren: boolean;
  expanded: boolean;
}

export function graphNodes(
  visible: ModelComponent[],
  all: ModelComponent[],
  expanded: Set<string>,
  selectedId: string | null,
): Node<GraphNodeData>[] {
  const rows = new Map<number, number>();
  const parents = new Set(all.map((item) => item.parent_id));
  return visible.map((component) => {
    const depth = component.id.split(".").length - 1;
    const row = rows.get(depth) ?? 0;
    rows.set(depth, row + 1);
    return {
      id: component.id,
      type: "module",
      // Explicit size: the MiniMap only draws nodes whose user object has dimensions.
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      position: { x: 320 * depth + 40, y: 96 * row + 40 },
      selected: component.id === selectedId,
      data: {
        id: component.id,
        moduleType: component.module_type,
        parameterCount: component.parameter_count,
        hasChildren: parents.has(component.id),
        expanded: expanded.has(component.id),
      },
    };
  });
}

export function graphEdges(components: ModelComponent[]): Edge[] {
  const ids = new Set(components.map((item) => item.id));
  return components.flatMap((component) =>
    component.parent_id !== null && ids.has(component.parent_id)
      ? [
          {
            id: `${component.parent_id}-${component.id}`,
            source: component.parent_id,
            target: component.id,
            type: "smoothstep",
            markerEnd: { type: MarkerType.ArrowClosed, color: "var(--color-strong)" },
            style: { stroke: "var(--color-strong)", strokeWidth: 1.5 },
          },
        ]
      : [],
  );
}

export function toggleSet(values: Set<string>, value: string): Set<string> {
  const next = new Set(values);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export function formatParameterCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
  return String(count);
}
