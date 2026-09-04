import { lazy, Suspense } from "react";

import type { ChartTheme } from "@/shared/chart";
import type { ModelComponent } from "@/shared/types";
import { Caption, EmptyState, Sheet, Skeleton } from "@/shared/ui";

import { formatParameterCount } from "./graph";

const HistogramCard = lazy(() => import("./HistogramCard").then((module) => ({ default: module.HistogramCard })));

interface MonitorInspectorProps {
  runId: string;
  component: ModelComponent;
  names: string[];
  running: boolean;
  chart: ChartTheme;
  onClose: () => void;
}

export function MonitorInspector({ runId, component, names, running, chart, onClose }: MonitorInspectorProps) {
  const available = new Set(names);
  const parameterNames = [...component.parameter_names]
    .sort((left, right) => rank(left) - rank(right) || left.localeCompare(right))
    .slice(0, 2);
  const charts = parameterNames.flatMap((parameter) => {
    const path = `${component.id}.${parameter}`;
    return [`param/${path}`, `grad/${path}`].filter((name) => available.has(name));
  });

  return (
    <Sheet open onClose={onClose} title={component.module_type} width={440}>
      <div className="flex flex-col gap-3 p-3">
        <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
          <Caption>Path</Caption>
          <dd className="m-0 truncate font-mono text-fg" title={component.id}>{component.id}</dd>
          <Caption>Parameters</Caption>
          <dd className="m-0 font-mono tabular-nums text-fg">{formatParameterCount(component.parameter_count)}</dd>
          <Caption>Tensors</Caption>
          <dd className="m-0 font-mono text-fg-secondary">{component.parameter_names.join(", ") || "—"}</dd>
        </dl>
        {charts.length === 0 ? (
          <EmptyState compact icon={<span />} title="No histograms" description="This module has not logged parameter or gradient histograms." />
        ) : (
          <Suspense fallback={<Skeleton className="h-64" />}>
            {charts.map((name) => (
              <HistogramCard key={name} runId={runId} name={name} running={running} chart={chart} />
            ))}
          </Suspense>
        )}
      </div>
    </Sheet>
  );
}

function rank(name: string) {
  if (name === "bias") return 0;
  if (name === "weight") return 1;
  return 2;
}
