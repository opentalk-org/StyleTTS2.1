import { Braces, Network, ScrollText } from "lucide-react";
import { lazy, Suspense, useState } from "react";

import { LogsPanel } from "@/features/logs/LogsPanel";
import { useVisibleUpdates } from "@/features/updates/query";
import type { ChartTheme } from "@/shared/chart";
import type { JsonValue, Run } from "@/shared/types";
import { Dialog, Skeleton, StatusBadge, Tabs, type TabItem } from "@/shared/ui";

import { JsonView } from "./JsonView";
import { useRunConfigQuery } from "./query";

const ModelMonitor = lazy(() =>
  import("@/features/model-monitor/ModelMonitor").then((module) => ({ default: module.ModelMonitor })),
);

type InspectorTab = "config" | "logs" | "graph";

const TABS: TabItem<InspectorTab>[] = [
  { id: "config", label: "Config", icon: <Braces /> },
  { id: "logs", label: "Logs", icon: <ScrollText /> },
  { id: "graph", label: "Model graph", icon: <Network /> },
];

interface RunInspectorProps {
  run: Run;
  color: string;
  chart: ChartTheme;
  onClose: () => void;
}

export function RunInspector({ run, color, chart, onClose }: RunInspectorProps) {
  const [tab, setTab] = useState<InspectorTab>("config");
  const config = useRunConfigQuery(run.id);
  const liveTab = tab === "logs" ? "run-logs" : "run-graph";
  useVisibleUpdates({
    projectId: run.projectId,
    runIds: [run.id],
    tab: liveTab,
    watchLineage: false,
    enabled: tab !== "config",
  });

  return (
    <Dialog open onClose={onClose} title={run.name} eyebrow={<StatusBadge status={run.status} />} size="viewport">
      <Tabs label="Run details" items={TABS} value={tab} onValue={setTab} />
      {tab === "config" ? (
        config.isPending ? <Skeleton className="m-4 flex-1" /> : config.isError ? (
          <p className="p-4 text-sm text-failed">{config.error.message}</p>
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-2 divide-x divide-line overflow-auto">
            <ConfigSection title="Training config" value={config.data.training} />
            <ConfigSection title="Data config" value={config.data.data} />
          </div>
        )
      ) : null}
      {tab === "logs" ? <LogsPanel runs={[run]} runColors={{ [run.id]: color }} /> : null}
      {tab === "graph" ? (
        <Suspense fallback={<Skeleton className="m-4 flex-1" />}>
          <ModelMonitor run={run} chart={chart} />
        </Suspense>
      ) : null}
    </Dialog>
  );
}

function ConfigSection({ title, value }: { title: string; value: JsonValue }) {
  return (
    <section className="min-w-0 overflow-auto">
      <h3 className="sticky top-0 z-10 border-b border-line bg-surface px-4 py-2 text-xs font-semibold uppercase tracking-wide text-fg-muted">
        {title}
      </h3>
      <JsonView value={value} />
    </section>
  );
}
