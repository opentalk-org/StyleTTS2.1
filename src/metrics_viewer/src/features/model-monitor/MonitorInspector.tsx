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
  loading: boolean;
  running: boolean;
  chart: ChartTheme;
  onClose: () => void;
}

export function MonitorInspector({ runId, component, names, loading, running, chart, onClose }: MonitorInspectorProps) {
  const available = new Set(names);
  const prefix = `param/${component.module_path ?? component.id}.`;
  const parameterPaths = names
    .filter((name) => name.startsWith(prefix))
    .map((name) => name.slice("param/".length))
    .sort((left, right) => {
      const leftName = left.slice(left.lastIndexOf(".") + 1);
      const rightName = right.slice(right.lastIndexOf(".") + 1);
      return rank(leftName) - rank(rightName) || left.localeCompare(right);
    })
    .slice(0, 2);
  const charts = parameterPaths.flatMap((path) => {
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
          <dd className="m-0 font-mono text-fg-secondary">
            {component.parameter_names.map((name) => `${name}: ${component.parameter_shapes?.[name] ?? "shape unavailable"}`).join(", ") || "—"}
          </dd>
          <Caption>Inputs</Caption>
          <dd className="m-0 font-mono text-fg-secondary">{component.input_shapes?.join(", ") || "—"}</dd>
          <Caption>Outputs</Caption>
          <dd className="m-0 font-mono text-fg-secondary">{component.output_shapes?.join(", ") || "—"}</dd>
        </dl>
        {loading ? (
          <Skeleton className="h-64" />
        ) : charts.length === 0 ? (
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
