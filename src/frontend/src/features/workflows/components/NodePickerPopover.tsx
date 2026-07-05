import { Icon } from "@/shared/icons";
import { addNode } from "../logic";
import { useWorkflowStore } from "../store";

export function NodePickerPopover({ onClose }: { onClose: () => void }) {
  const { schema, graph, setGraph } = useWorkflowStore();
  if (!schema) return null;
  const nodes = Object.values(schema.nodes).sort((a, b) => a.category.localeCompare(b.category) || a.type.localeCompare(b.type));
  return (
    <div className="absolute bottom-14 left-4 z-20 max-h-[420px] w-[320px] overflow-y-auto rounded-md border border-line bg-panel p-2 shadow-xl">
      {nodes.map((node) => (
        <button
          key={node.type}
          onClick={() => {
            setGraph(addNode(schema, graph, node.type, 80 + graph.nodes.length * 24, 120 + graph.nodes.length * 12));
            onClose();
          }}
          className="flex h-10 w-full items-center gap-2 rounded-md px-2 text-left hover:bg-panel-2"
        >
          <Icon name="plus" size={15} className="text-txt-mute" />
          <span className="flex-1 text-[13px] font-semibold text-txt">{node.type}</span>
          <span className="text-[11px] text-txt-mute">{node.category}</span>
        </button>
      ))}
    </div>
  );
}
