import { Check, SlidersHorizontal } from "lucide-react";

import type { Run } from "@/shared/types";
import { Badge, Card, cn, EmptyState, GroupLabel } from "@/shared/ui";

export function ParamsPanel({ runs }: { runs: Run[] }) {
  if (runs.length === 0) {
    return (
      <EmptyState
        icon={<SlidersHorizontal />}
        title="No runs selected"
        description="Select runs to line up their hyperparameters and see which values differ."
      />
    );
  }

  const names = [...new Set(runs.flatMap((run) => Object.keys(run.params)))];
  const template = `180px repeat(${runs.length}, minmax(160px, 1fr))`;

  return (
    <Card className="overflow-auto">
      <div className="min-w-[700px]">
        <div
          className="sticky top-0 z-10 grid border-b border-line bg-inset"
          style={{ gridTemplateColumns: template }}
        >
          <GroupLabel className="border-r border-line px-3.5 py-3">Parameter</GroupLabel>
          {runs.map((run) => (
            <GroupLabel key={run.id} className="truncate border-r border-line px-3.5 py-3 last:border-r-0">
              {run.name}
            </GroupLabel>
          ))}
        </div>
        {names.map((name) => {
          const values = runs.map((run) => String(run.params[name]));
          const identical = new Set(values).size === 1;
          return (
            <div
              key={name}
              className={cn(
                "grid border-b border-line last:border-b-0",
                identical ? "" : "bg-accent-surface/40",
              )}
              style={{ gridTemplateColumns: template }}
            >
              <div className="flex items-center justify-between gap-2 border-r border-line px-3.5 py-3">
                <span className="truncate font-mono text-xs text-fg">{name}</span>
                {identical ? (
                  <Check size={12} className="shrink-0 text-fg-muted" aria-label="identical across runs" />
                ) : (
                  <Badge tone="accent" className="shrink-0 px-1.5 font-mono text-[10px]">
                    diff
                  </Badge>
                )}
              </div>
              {values.map((value, index) => (
                <div
                  key={runs[index].id}
                  className={cn(
                    "truncate border-r border-line px-3.5 py-3 font-mono text-xs tabular-nums last:border-r-0",
                    identical ? "text-fg-secondary" : "text-fg",
                  )}
                >
                  {value}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </Card>
  );
}
