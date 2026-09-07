import { lazy, Suspense } from "react";

import type { ChartTheme } from "@/shared/chart";
import { Caption, EmptyState, Sheet, Skeleton } from "@/shared/ui";

import { formatParameterCount } from "./hierarchy";
import type { LevelNode } from "./level";

const HistogramCard = lazy(() => import("./HistogramCard").then((module) => ({ default: module.HistogramCard })));

interface MonitorInspectorProps {
  runId: string;
  box: LevelNode;
  names: string[];
  loading: boolean;
  running: boolean;
  chart: ChartTheme;
  onClose: () => void;
}

export function MonitorInspector({ runId, box, names, loading, running, chart, onClose }: MonitorInspectorProps) {
  const component = box.node.component;
  const charts = histogramNames(names, component?.module_path ?? box.node.id);

  return (
    <Sheet open onClose={onClose} title={box.label.moduleType} width={440}>
      <div className="flex flex-col gap-3 p-3">
        <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
          <Caption>Path</Caption>
          <dd className="m-0 truncate font-mono text-fg" title={box.node.id}>{box.node.id}</dd>
          <Caption>Parameters</Caption>
          <dd className="m-0 font-mono tabular-nums text-fg">{formatParameterCount(box.parameterCount)}</dd>
          <Caption>Invocations</Caption>
          <dd className="m-0 font-mono tabular-nums text-fg-secondary">
            {box.invocationCount}
            {box.repeatCount > 1 ? ` in ${box.repeatCount} repeats of this block` : ""}
          </dd>
          {component === null || component.input_ids === undefined ? null : (
            <>
              <Caption>Tensors</Caption>
              <dd className="m-0 font-mono text-fg-secondary">
                {component.parameter_names
                  .map((name) => `${name}: ${component.parameter_shapes?.[name] ?? "shape unavailable"}`)
                  .join(", ") || "—"}
              </dd>
              <Caption>Inputs</Caption>
              <dd className="m-0 font-mono text-fg-secondary">{component.input_shapes?.join(", ") || "—"}</dd>
              <Caption>Outputs</Caption>
              <dd className="m-0 font-mono text-fg-secondary">{component.output_shapes?.join(", ") || "—"}</dd>
            </>
          )}
        </dl>
        {loading ? (
          <Skeleton className="h-64" />
        ) : charts.length === 0 ? (
          <EmptyState
            compact
            icon={<span />}
            title="No histograms"
            description="No parameter or gradient histogram was logged under this path."
          />
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

/** The two most telling tensors under a path, each paired with its gradient series. */
function histogramNames(names: string[], path: string): string[] {
  const available = new Set(names);
  const prefix = `param/${path}.`;
  return names
    .filter((name) => name.startsWith(prefix))
    .map((name) => name.slice("param/".length))
    .sort((left, right) => {
      const leftName = left.slice(left.lastIndexOf(".") + 1);
      const rightName = right.slice(right.lastIndexOf(".") + 1);
      return rank(leftName) - rank(rightName) || left.localeCompare(right);
    })
    .slice(0, 2)
    .flatMap((name) => [`param/${name}`, `grad/${name}`].filter((series) => available.has(series)));
}

function rank(name: string) {
  if (name === "bias") return 0;
  if (name === "weight") return 1;
  return 2;
}
