import { Handle, Position, type NodeProps } from "@xyflow/react";
import { ChevronRight } from "lucide-react";
import { createContext, useContext } from "react";

import { cn } from "@/shared/ui";

import type { GraphNodeData } from "./graph";
import { formatParameterCount } from "./hierarchy";
import { hueColor, hueLine, hueSurface } from "./palette";

export interface GraphActionSet {
  /** Draw a module's children inside it, or fold them back away. */
  setOpen: (id: string, open: boolean) => void;
  /** Draw every block of a repeat run separately, or fold the run back into one box. */
  setUnfolded: (id: string, unfolded: boolean) => void;
}

/** Opening a box relays out the canvas, so nodes report the intent instead of holding state. */
export const GraphActions = createContext<GraphActionSet>({ setOpen: () => undefined, setUnfolded: () => undefined });

export function ModuleNode({ data, selected }: NodeProps & { data: GraphNodeData }) {
  return (
    <div
      style={{
        background: hueSurface(data.hue),
        borderColor: selected || data.linked ? "var(--color-accent)" : hueLine(data.hue, 55),
        opacity: data.dimmed ? 0.3 : 1,
      }}
      className={cn(
        "flex size-full flex-col justify-center gap-0.5 rounded-lg border px-3 py-1.5",
        selected ? "outline-2 outline-accent/40" : "",
      )}
    >
      <Handle type="target" position={Position.Top} className="!size-1.5 !border-0 !bg-strong" />
      <span className="truncate text-[13px] font-semibold text-fg">{data.moduleType || data.name}</span>
      <div className="flex items-center justify-between gap-2">
        <span className="truncate font-mono text-[11px] text-fg-muted">
          {data.name}
          <Trail chain={data.chain} />
        </span>
        <span className="flex shrink-0 items-center gap-1.5">
          <RepeatBadge data={data} />
          <span className="font-mono text-[11px] tabular-nums text-fg-muted">
            {formatParameterCount(data.parameterCount)}
          </span>
          <Opener data={data} />
        </span>
      </div>
      <Handle type="source" position={Position.Bottom} className="!size-1.5 !border-0 !bg-strong" />
    </div>
  );
}

export function ContainerNode({ data, selected }: NodeProps & { data: GraphNodeData }) {
  const { setOpen } = useContext(GraphActions);
  return (
    <div
      style={{
        borderColor: selected || data.linked ? "var(--color-accent)" : hueColor(data.hue),
        opacity: data.dimmed ? 0.5 : 1,
      }}
      className="size-full rounded-xl border-[1.5px]"
    >
      <button
        type="button"
        title={`Fold ${data.id} away`}
        onClick={(event) => {
          event.stopPropagation();
          setOpen(data.id, false);
        }}
        style={{ background: hueColor(data.hue) }}
        className="pointer-events-auto absolute -top-2.5 left-3 flex max-w-[calc(100%-24px)] items-center gap-1.5 rounded-full px-2 py-0.5"
      >
        <span className="truncate text-[11px] font-semibold text-accent-fg">{data.moduleType || data.name}</span>
        {data.moduleType.length === 0 ? null : (
          <span className="truncate font-mono text-[11px] text-accent-fg/70">{data.name}</span>
        )}
        <RepeatBadge data={data} />
      </button>
    </div>
  );
}

/** Wrapper containers this box swallowed, so the path it stands for is still readable. */
function Trail({ chain }: { chain: string[] }) {
  if (chain.length === 0) return null;
  return <span className="text-fg-disabled"> › {chain.join(" › ")}</span>;
}

function Opener({ data }: { data: GraphNodeData }) {
  const { setOpen } = useContext(GraphActions);
  if (!data.container) return null;
  return (
    <button
      type="button"
      title={`Open ${data.id}`}
      onClick={(event) => {
        event.stopPropagation();
        setOpen(data.id, true);
      }}
      className="flex items-center rounded-sm font-mono text-[11px] text-fg-muted hover:bg-hover hover:text-fg"
    >
      <ChevronRight size={11} />
      {data.invocationCount}
    </button>
  );
}

/** The count of a repeat run, and the control that unfolds it into its individual blocks. */
function RepeatBadge({ data }: { data: GraphNodeData }) {
  const { setUnfolded } = useContext(GraphActions);
  if (data.repeatCount < 2) return null;
  return (
    <button
      type="button"
      title={data.unfolded ? `Fold ${data.repeatCount} identical blocks back into one` : `Show all ${data.repeatCount} blocks separately`}
      onClick={(event) => {
        event.stopPropagation();
        setUnfolded(data.id, !data.unfolded);
      }}
      className={cn(
        "shrink-0 rounded-sm px-0.5 font-mono text-[11px] font-semibold",
        data.open ? "text-accent-fg/80 hover:bg-black/10" : "text-fg-secondary hover:bg-hover hover:text-fg",
        data.unfolded ? "underline" : "",
      )}
    >
      ×{data.repeatCount}
    </button>
  );
}
