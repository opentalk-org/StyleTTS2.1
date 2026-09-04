import { AlertCircle, ChevronDown, ChevronRight, Play, RotateCcw } from "lucide-react";
import { useState } from "react";

import { isDefaultSql, useViewerStore } from "@/features/viewer/store";
import type { PlotQueryResult } from "@/shared/types";
import { Badge, Button, Sheet, Textarea } from "@/shared/ui";

interface QuerySheetProps {
  open: boolean;
  onClose: () => void;
  onRun: () => void;
  running: boolean;
  error: Error | null;
  result: PlotQueryResult | null;
  plotCount: number;
}

export function QuerySheet({ open, onClose, onRun, running, error, result, plotCount }: QuerySheetProps) {
  const { sql, runningSql, setSql, resetSql } = useViewerStore();
  const [helpOpen, setHelpOpen] = useState(false);
  const dirty = sql !== runningSql;

  return (
    <Sheet
      open={open}
      onClose={onClose}
      title="Chart query"
      width={460}
      actions={
        <>
          {isDefaultSql(sql) ? null : (
            <Button size="sm" variant="ghost" icon={<RotateCcw size={12} />} onClick={resetSql}>
              Reset
            </Button>
          )}
          <Button size="sm" variant="primary" icon={<Play size={11} fill="currentColor" />} disabled={running} onClick={onRun}>
            {running ? "Running…" : "Run"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3 p-3">
        <div className="flex items-center gap-2 text-xs text-fg-muted">
          {dirty ? <Badge tone="notice">Edited, not run</Badge> : null}
          {!dirty && result !== null ? (
            <span className="font-mono tabular-nums">
              {plotCount} charts · {result.x.length.toLocaleString()} points · {result.elapsedMs} ms
            </span>
          ) : null}
          <span className="ml-auto">⌘⏎ to run</span>
        </div>
        <Textarea
          value={sql}
          onChange={(event) => setSql(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              if (!running) onRun();
            }
          }}
          autoGrow
          className="min-h-48"
          aria-label="Chart query"
        />
        {error === null ? null : (
          <div role="alert" className="flex items-start gap-2 rounded-md bg-failed-bg p-3 text-failed">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <div className="min-w-0 text-xs leading-relaxed">
              <strong className="block font-medium">Query failed</strong>
              <span className="break-words opacity-90">{error.message}</span>
            </div>
          </div>
        )}
        <button
          type="button"
          aria-expanded={helpOpen}
          onClick={() => setHelpOpen(!helpOpen)}
          className="flex items-center gap-1 text-xs font-medium text-fg-muted hover:text-fg"
        >
          {helpOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          Reference
        </button>
        {helpOpen ? <QueryReference /> : null}
      </div>
    </Sheet>
  );
}

function QueryReference() {
  return (
    <div className="flex flex-col gap-2 text-xs leading-relaxed text-fg-muted">
      <p>
        Plain ClickHouse SQL. Return one column <Token>AS plot</Token> (one chart per distinct value), one{" "}
        <Token>AS x</Token>, one <Token>AS y</Token>, and <Token>run_id</Token> to draw one line per run.
      </p>
      <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1">
        <dt className="font-mono text-fg-secondary">metrics</dt>
        <dd className="m-0">run_id, name, step, timestamp, value</dd>
        <dt className="font-mono text-fg-secondary">{"{run_ids:Array(UUID)}"}</dt>
        <dd className="m-0">The selected runs</dd>
        <dt className="font-mono text-fg-secondary">{"{project_id:UUID}"}</dt>
        <dd className="m-0">The open project</dd>
        <dt className="font-mono text-fg-secondary">largestTriangleThreeBuckets(n)(x, y)</dt>
        <dd className="m-0">Downsample long runs in the database</dd>
      </dl>
      <p>
        The default query plots every metric downsampled to 1000 points per series and also returns{" "}
        <Token>wall</Token> (epoch ms) and <Token>rel</Token> (seconds since the run started) so charts can
        switch x axis. Narrow it with <Token>AND name IN [...]</Token>, or raise the bucket count for more
        detail. A custom query without those two columns can only plot against its own <Token>x</Token>.
      </p>
    </div>
  );
}

function Token({ children }: { children: string }) {
  return <code className="rounded-sm bg-hover px-1 py-0.5 font-mono text-[11px] text-fg-secondary">{children}</code>;
}
