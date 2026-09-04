import { X } from "lucide-react";

import type { ModelComponent } from "@/shared/types";
import { IconButton } from "@/shared/ui";
import { HistogramCard } from "./HistogramCard";

export function MonitorInspector({ runId, component, names, running, onClose }: {
  runId: string;
  component: ModelComponent;
  names: string[];
  running: boolean;
  onClose: () => void;
}) {
  const available = new Set(names);
  const parameterNames = [...component.parameter_names].sort((left, right) =>
    rank(left) - rank(right) || left.localeCompare(right),
  ).slice(0, 2);
  const charts = parameterNames.flatMap((parameter) => {
    const path = `${component.id}.${parameter}`;
    return [`param/${path}`, `grad/${path}`].filter((name) => available.has(name));
  });

  return (
    <aside className="absolute top-15 bottom-3 left-3 z-20 flex w-[min(840px,calc(100%-1.5rem))] flex-col overflow-hidden rounded-lg border border-line bg-elevated shadow-2xl">
      <header className="flex h-12 flex-none items-center justify-between border-b border-line px-3">
        <h2 className="m-0 text-sm font-semibold text-fg">
          {component.module_type} <span className="font-mono text-xs font-normal text-fg-muted">{component.id}</span>
        </h2>
        <IconButton label="Close" onClick={onClose}><X size={16} /></IconButton>
      </header>
      <div className="grid min-h-0 grid-cols-1 gap-2 overflow-auto p-2 sm:grid-cols-2 sm:grid-rows-2">
        {charts.map((name) => <HistogramCard key={name} runId={runId} name={name} running={running} />)}
      </div>
    </aside>
  );
}

function rank(name: string) {
  if (name === "bias") return 0;
  if (name === "weight") return 1;
  return 2;
}
