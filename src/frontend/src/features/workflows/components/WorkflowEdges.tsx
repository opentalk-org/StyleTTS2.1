import { useWorkflowStore } from "../store";
import type { PortAnchorKey } from "../types";

export function WorkflowEdges() {
  const { graph, schema, viewport, wireDraft, portAnchors } = useWorkflowStore();
  const nodeById = Object.fromEntries(graph.nodes.map((node) => [node.id, node]));
  const project = (x: number, y: number) => ({ x: x * viewport.zoom + viewport.x, y: y * viewport.zoom + viewport.y });
  const anchor = (nodeId: string, portName: string, kind: "input" | "output") => portAnchors[`${nodeId}:${portName}:${kind}` as PortAnchorKey];
  const path = (start: { x: number; y: number }, end: { x: number; y: number }) => {
    const dx = Math.max(20, Math.abs(end.x - start.x) * 0.25);
    return `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
  };
  const wireNode = wireDraft ? nodeById[wireDraft.node] : undefined;
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full">
      {graph.edges.map((edge) => {
        const source = nodeById[edge.source_node];
        const target = nodeById[edge.target_node];
        if (!source || !target) return null;
        const start = anchor(edge.source_node, edge.source_port, "output") ?? project(source.x + 220, source.y + 92);
        const end = anchor(edge.target_node, edge.target_port, "input") ?? project(target.x, target.y + 92);
        const sourceInfo = schema?.nodes[source.type];
        const port = sourceInfo?.outputs[edge.source_port];
        const schemaType = port && schema ? schema.types[port.type] : undefined;
        const color = schemaType ? schemaType.color : "#6b7280";
        return <path key={`${edge.source_node}-${edge.source_port}-${edge.target_node}-${edge.target_port}`} d={path(start, end)} fill="none" stroke={color} strokeWidth={2} />;
      })}
      {wireDraft && wireNode ? <WirePreview node={wireNode} wire={wireDraft} anchor={anchor} project={project} path={path} /> : null}
    </svg>
  );
}

function WirePreview({
  node,
  wire,
  anchor,
  project,
  path,
}: {
  node: { id: string; x: number; y: number };
  wire: { port: string; kind: "input" | "output"; x: number; y: number };
  anchor: (nodeId: string, portName: string, kind: "input" | "output") => { x: number; y: number } | undefined;
  project: (x: number, y: number) => { x: number; y: number };
  path: (start: { x: number; y: number }, end: { x: number; y: number }) => string;
}) {
  const socket = anchor(node.id, wire.port, wire.kind) ?? project(node.x + (wire.kind === "output" ? 220 : 0), node.y + 92);
  const cursor = project(wire.x, wire.y);
  const start = wire.kind === "output" ? socket : cursor;
  const end = wire.kind === "output" ? cursor : socket;
  return <path d={path(start, end)} fill="none" stroke="#2563eb" strokeDasharray="6 6" strokeWidth={2} />;
}
