import { useVirtualizer } from "@tanstack/react-virtual";
import { ScrollText } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";

import type { Run, RunLog } from "@/shared/types";
import { Button, EmptyState } from "@/shared/ui";

import { useLogsQuery } from "./query";

interface LogsPanelProps {
  runs: Run[];
}

const NODE_PREFIX_PATTERN = /^\[[^\]]+\] /;

export function LogsPanel({ runs }: LogsPanelProps) {
  const runIds = useMemo(() => runs.map((run) => run.id).sort(), [runs]);
  const logs = useLogsQuery(runIds, runs.length > 0);
  const rows = useMemo(() => logs.data?.pages.flatMap((page) => page.rows) ?? [], [logs.data]);
  const scroll = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scroll.current,
    estimateSize: () => 52,
    overscan: 12,
  });
  const virtualRows = virtual.getVirtualItems();
  const lastIndex = virtualRows.at(-1)?.index ?? -1;

  useEffect(() => {
    if (lastIndex < rows.length - 20 || !logs.hasNextPage || logs.isFetchingNextPage) return;
    void logs.fetchNextPage();
  }, [lastIndex, rows.length, logs.hasNextPage, logs.isFetchingNextPage, logs.fetchNextPage]);

  if (runs.length === 0) {
    return <EmptyState icon={<ScrollText />} title="Select runs to see logs" description="Logs from the selected runs appear here as they are written." />;
  }
  if (logs.isPending) return <LogLoading />;
  if (logs.isError) {
    return <EmptyState icon={<ScrollText />} title="Could not load logs" description={logs.error.message} />;
  }
  if (rows.length === 0) {
    return <EmptyState icon={<ScrollText />} title="No logs recorded" description="The selected runs have not written any ClickHouse logs." />;
  }

  return (
    <div ref={scroll} className="min-h-0 flex-1 overflow-auto bg-inset">
      <div className="relative w-full" style={{ height: virtual.getTotalSize() }}>
        {virtualRows.map((item) => {
          const row = rows[item.index];
          return (
            <LogRow
              key={logKey(row)}
              ref={virtual.measureElement}
              row={row}
              index={item.index}
              offset={item.start}
            />
          );
        })}
      </div>
      {logs.hasNextPage ? (
        <div className="sticky bottom-0 flex justify-center border-t border-line bg-surface/95 p-2">
          <Button size="sm" disabled={logs.isFetchingNextPage} onClick={() => void logs.fetchNextPage()}>
            {logs.isFetchingNextPage ? "Loading…" : "Load older logs"}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

interface LogRowProps {
  ref: (element: Element | null) => void;
  row: RunLog;
  index: number;
  offset: number;
}

function LogRow({ ref, row, index, offset }: LogRowProps) {
  return (
    <article
      ref={ref}
      data-index={index}
      className="absolute left-0 top-0 grid w-full grid-cols-[170px_minmax(0,1fr)] border-b border-line px-3 py-2 font-mono text-xs leading-5"
      style={{ transform: `translateY(${offset}px)` }}
    >
      <time className="tabular-nums text-fg-muted">{new Date(row.timestamp).toLocaleString()}</time>
      <pre className="m-0 whitespace-pre-wrap break-words font-mono text-xs text-fg">
        {row.message.replace(NODE_PREFIX_PATTERN, "")}
      </pre>
    </article>
  );
}

function LogLoading() {
  return <div className="flex min-h-0 flex-1 items-center justify-center text-sm text-fg-muted">Loading logs…</div>;
}

function logKey(log: RunLog) {
  return `${log.runId}:${log.timestamp}:${log.message}`;
}
