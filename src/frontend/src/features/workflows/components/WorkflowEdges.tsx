import { useWorkflowStore } from "../store";

export function WorkflowEdges() {
  const { graph, schema, viewport, wireDraft } = useWorkflowStore();
  const nodeById = Object.fromEntries(graph.nodes.map((node) => [node.id, node]));
  const project = (x: number, y: number) => ({ x: x * viewport.zoom + viewport.x, y: y * viewport.zoom + viewport.y });
  const path = (start: { x: number; y: number }, end: { x: number; y: number }) => {
    const dx = Math.max(60, Math.abs(end.x - start.x) * 0.5);
    return `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
  };
  const wireSource = wireDraft ? nodeById[wireDraft.source_node] : undefined;
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full">
      {graph.edges.map((edge) => {
        const source = nodeById[edge.source_node];
        const target = nodeById[edge.target_node];
        if (!source || !target) return null;
        const start = project(source.x + 240, source.y + 90);
        const end = project(target.x, target.y + 90);
        const sourceInfo = schema?.nodes[source.type];
        const port = sourceInfo?.outputs[edge.source_port];
        const schemaType = port && schema ? schema.types[port.type] : undefined;
        const color = schemaType ? schemaType.color : "#6b7280";
        return <path key={`${edge.source_node}-${edge.source_port}-${edge.target_node}-${edge.target_port}`} d={path(start, end)} fill="none" stroke={color} strokeWidth={2} />;
      })}
      {wireDraft && wireSource ? <WirePreview source={wireSource} target={wireDraft} project={project} path={path} /> : null}
    </svg>
  );
}

function WirePreview({
  source,
  target,
  project,
  path,
}: {
  source: { x: number; y: number };
  target: { x: number; y: number };
  project: (x: number, y: number) => { x: number; y: number };
  path: (start: { x: number; y: number }, end: { x: number; y: number }) => string;
}) {
  const start = project(source.x + 240, source.y + 90);
  return <path d={path(start, project(target.x, target.y))} fill="none" stroke="#2563eb" strokeDasharray="6 6" strokeWidth={2} />;
}
