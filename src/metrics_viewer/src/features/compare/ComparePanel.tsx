import { ChartScatter, Columns3, Plus, Waypoints } from "lucide-react";
import { lazy, Suspense, useMemo, useState } from "react";

import { useViewerStore } from "@/features/viewer/store";
import type { ChartTheme } from "@/shared/chart";
import { projectColumnOptions } from "@/shared/metrics";
import type { ProjectColumns, Run } from "@/shared/types";
import { COLUMN_CLASSES, type Columns as GridColumns } from "@/features/comparison/ChartSection";
import { Button, Checkbox, cn, Popover, SearchOptionList, SegmentedControl, Toolbar } from "@/shared/ui";

import { CompareTable } from "./CompareTable";
import { defaultCompareColumns } from "./logic";

const ComparePlot = lazy(() => import("./ComparePlot").then((module) => ({ default: module.ComparePlot })));
const ParallelPlot = lazy(() => import("./ParallelPlot").then((module) => ({ default: module.ParallelPlot })));

interface ComparePanelProps {
  projectColumns: ProjectColumns;
  /** Runs ticked in the run list; these are the table rows. */
  runs: Run[];
  runColors: Record<string, string>;
  chart: ChartTheme;
}

export function ComparePanel({ projectColumns, runs, runColors, chart }: ComparePanelProps) {
  const compare = useViewerStore((state) => state.compare);
  const setCompareColumns = useViewerStore((state) => state.setCompareColumns);
  const addComparePlot = useViewerStore((state) => state.addComparePlot);
  const updateComparePlot = useViewerStore((state) => state.updateComparePlot);
  const removeComparePlot = useViewerStore((state) => state.removeComparePlot);
  const [onlyDifferences, setOnlyDifferences] = useState(false);
  const [columnsOpen, setColumnsOpen] = useState(false);
  const [grid, setGrid] = useState<GridColumns>("auto");

  const columns = useMemo(() => compare.columns ?? defaultCompareColumns(runs), [compare.columns, runs]);
  const columnOptions = useMemo(() => projectColumnOptions(projectColumns), [projectColumns]);

  return (
    <div className="flex min-h-0 flex-1 flex-row">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Toolbar
          start={
            <>
              <Popover
                open={columnsOpen}
                onClose={() => setColumnsOpen(false)}
                align="start"
                width={320}
                panelClassName="[&>div]:p-0"
                trigger={
                  <Button variant="secondary" icon={<Columns3 size={14} />} aria-expanded={columnsOpen} onClick={() => setColumnsOpen(!columnsOpen)}>
                    Columns
                    <span className="font-mono text-[11px] text-fg-muted">{columns.length}</span>
                  </Button>
                }
              >
                <SearchOptionList
                  multiple
                  options={columnOptions}
                  selected={columns}
                  placeholder="Search parameters and metrics"
                  emptyMessage="No column matches"
                  onSelect={(id) =>
                    setCompareColumns(columns.includes(id) ? columns.filter((column) => column !== id) : [...columns, id])
                  }
                />
              </Popover>
              {compare.columns === null ? null : (
                <Button variant="ghost" size="sm" onClick={() => setCompareColumns(null)}>
                  Default columns
                </Button>
              )}
              <Checkbox checked={onlyDifferences} onChange={(event) => setOnlyDifferences(event.target.checked)}>
                Only differences
              </Checkbox>
            </>
          }
          end={
            <>
              <span className="font-mono text-xs tabular-nums text-fg-muted">{runs.length} runs</span>
              <SegmentedControl
                label="Plot columns"
                value={grid}
                onValue={setGrid}
                options={[
                  { value: "1" as const, label: "1" },
                  { value: "2" as const, label: "2" },
                  { value: "3" as const, label: "3" },
                  { value: "auto" as const, label: "Auto" },
                ]}
              />
              <Button variant="secondary" icon={<Plus size={14} />} onClick={() => addComparePlot("chart")}>
                Add plot
                {compare.plots.length > 0 ? <ChartScatter size={13} className="text-fg-muted" /> : null}
              </Button>
              <Button variant="secondary" icon={<Waypoints size={14} />} onClick={() => addComparePlot("parallel")}>
                Parallel
              </Button>
            </>
          }
        />
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="flex flex-col gap-2 p-2">
            {compare.plots.length > 0 ? (
              <div className={cn("grid gap-1.5", COLUMN_CLASSES[grid])}>
                <Suspense fallback={null}>
                  {compare.plots.map((plot) =>
                    plot.kind === "parallel" ? (
                      <ParallelPlot
                        key={plot.id}
                        runs={runs}
                        runColors={runColors}
                        columns={columns}
                        columnOptions={columnOptions}
                        config={plot}
                        onConfig={(patch) => updateComparePlot(plot.id, patch)}
                        onRemove={() => removeComparePlot(plot.id)}
                      />
                    ) : (
                      <ComparePlot
                        key={plot.id}
                        runs={runs}
                        runColors={runColors}
                        chart={chart}
                        config={plot}
                        onConfig={(patch) => updateComparePlot(plot.id, patch)}
                        onRemove={() => removeComparePlot(plot.id)}
                      />
                    ),
                  )}
                </Suspense>
              </div>
            ) : null}
            <div className="overflow-x-auto rounded-md border border-line bg-surface">
              <CompareTable runs={runs} columns={columns} runColors={runColors} onlyDifferences={onlyDifferences} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
