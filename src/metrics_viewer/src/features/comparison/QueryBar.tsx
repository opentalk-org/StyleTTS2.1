import { AlertCircle, ChevronDown, ChevronRight, Database, Play, RotateCcw } from "lucide-react";
import { useState } from "react";

import { METRIC_NAMES } from "@/shared/metrics";
import { Badge, Button, Card, cn, GroupLabel, IconButton, Textarea } from "@/shared/ui";

export interface QueryBarProps {
  sql: string;
  onSql: (sql: string) => void;
  onRun: () => void;
  running: boolean;
  error: Error | null;
  /** Result summary, shown once a query has succeeded. */
  summary: string | null;
  /** The editor holds edits that have not been run yet. */
  dirty: boolean;
  onReset: () => void;
}

/**
 * The query that defines the plots. It sits above the charts because editing it is
 * how you choose what is plotted — there is no separate chart builder.
 */
export function QueryBar({ sql, onSql, onRun, running, error, summary, dirty, onReset }: QueryBarProps) {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <div className="flex min-h-11 flex-wrap items-center justify-between gap-x-3 gap-y-2 px-3 py-2">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className="flex min-w-0 items-center gap-2 text-fg-muted transition-colors duration-150 hover:text-fg-secondary"
        >
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          <Database size={13} />
          <GroupLabel>Plots query</GroupLabel>
          {open ? null : (
            <span className="truncate font-mono text-[11px] text-fg-muted">
              {sql.replace(/\s+/g, " ").slice(0, 70)}…
            </span>
          )}
        </button>
        <div className="flex items-center gap-2">
          {dirty ? (
            <Badge tone="notice">Edited — not run</Badge>
          ) : summary === null ? null : (
            <span className="font-mono text-xs tabular-nums text-fg-muted">{summary}</span>
          )}
          <IconButton label="Restore the default query" size="sm" onClick={onReset}>
            <RotateCcw size={13} />
          </IconButton>
          <Button
            variant={dirty ? "primary" : "secondary"}
            size="sm"
            icon={<Play size={12} fill="currentColor" />}
            disabled={running}
            title="Run query (Ctrl/Cmd + Enter)"
            onClick={onRun}
          >
            {running ? "Running…" : "Run"}
          </Button>
        </div>
      </div>

      {open ? (
        <>
          <Textarea
            value={sql}
            onChange={(event) => onSql(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                if (!running) onRun();
              }
            }}
            autoGrow
            className="min-h-32 border-t border-line bg-inset"
            aria-label="Plots query"
          />
          <QueryHelp />
        </>
      ) : null}

      {error === null ? null : (
        <div
          role="alert"
          className={cn(
            "flex items-start gap-2.5 border-t border-negative-border bg-negative-surface p-3 text-negative",
          )}
        >
          <AlertCircle size={15} className="mt-px shrink-0" />
          <div className="min-w-0">
            <strong className="block text-xs font-medium">Query failed</strong>
            <p className="m-0 mt-1 text-xs leading-relaxed opacity-90">{error.message}</p>
          </div>
        </div>
      )}
    </Card>
  );
}

/** The contract is small enough to state in full, so it is stated in full. */
function QueryHelp() {
  return (
    <div className="flex flex-col gap-2 border-t border-line bg-elevated px-3 py-2.5">
      <GroupLabel>Contract</GroupLabel>
      <p className="m-0 text-xs leading-relaxed text-fg-muted">
        Alias one column <Token>AS plot</Token> (one chart per distinct value), one{" "}
        <Token>AS x</Token> and one <Token>AS y</Token>, and select <Token>run_id</Token> to
        split each chart into one line per run.
      </p>
      <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs text-fg-muted">
        <dt className="font-mono text-fg-secondary">metrics(names =&gt; [], runs =&gt; …)</dt>
        <dd className="m-0">Source rows: name, run_id, step, timestamp, value</dd>
        <dt className="font-mono text-fg-secondary">selected()</dt>
        <dd className="m-0">The runs ticked in the table</dd>
        <dt className="font-mono text-fg-secondary">all_runs()</dt>
        <dd className="m-0">Every run in the project</dd>
        <dt className="font-mono text-fg-secondary">WHERE x BETWEEN a AND b</dt>
        <dd className="m-0">Clip the x range</dd>
        <dt className="font-mono text-fg-secondary">LIMIT n</dt>
        <dd className="m-0">Cap the returned rows</dd>
      </dl>
      <p className="m-0 text-xs text-fg-muted">
        Metrics available: <span className="font-mono">{METRIC_NAMES.join(", ")}</span>
      </p>
    </div>
  );
}

function Token({ children }: { children: string }) {
  return (
    <code className="rounded bg-surface px-1 py-0.5 font-mono text-[11px] text-fg-secondary">
      {children}
    </code>
  );
}
