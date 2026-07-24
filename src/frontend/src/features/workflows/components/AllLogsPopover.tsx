import { useEffect, useMemo, useState } from "react";

import { copyText } from "@/shared/clipboard";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { fetchNodeLog, fetchRun, fetchRunErrors } from "../api";
import { useWorkflowStore } from "../store";
import type { RunErrorEvent, RunStatus } from "../types";

type LogEntry = { nodeId: string; content: string; truncated: boolean; error: string | null };
type LogRecord = { ts: number | null; seq: number; nodeId: string; text: string };

const TIMESTAMP = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})[.,](\d{3})/;

function parseTimestamp(line: string): number | null {
  const match = TIMESTAMP.exec(line);
  if (!match) return null;
  const value = Date.parse(`${match[1]}T${match[2]}.${match[3]}`);
  return Number.isNaN(value) ? null : value;
}

function toRecords(entry: LogEntry, seqStart: number): LogRecord[] {
  const records: LogRecord[] = [];
  let seq = seqStart;
  if (entry.error) records.push({ ts: null, seq: seq++, nodeId: entry.nodeId, text: `[error] ${entry.error}` });
  let current: LogRecord | null = null;
  for (const line of entry.content.split("\n")) {
    if (line === "" && current === null) continue;
    const ts = parseTimestamp(line);
    if (ts !== null || current === null) {
      current = { ts, seq: seq++, nodeId: entry.nodeId, text: line };
      records.push(current);
    } else {
      current.text += `\n${line}`;
    }
  }
  return records;
}

function nodeColor(index: number): string {
  return `hsl(${(index * 67) % 360} 70% 45%)`;
}

function runFailureEntry(status: RunStatus, errors: RunErrorEvent[]): LogEntry | null {
  const failure = [...errors].reverse().find((event) => event.kind === "run_failed");
  const message = failure ? failure.message : status.error;
  if (message === null) return null;
  const traceback = failure && typeof failure.detail.traceback === "string" ? failure.detail.traceback.trim() : "";
  const timestamp = failure?.created_at ?? status.finished_at ?? status.created_at;
  return {
    nodeId: "run",
    content: `${timestamp} ERROR [run] ${message}${traceback ? `\n${traceback}` : ""}`,
    truncated: false,
    error: null,
  };
}

function nodeFailureEntries(errors: RunErrorEvent[], loggedNodeIds: Set<string>): LogEntry[] {
  return errors
    .filter((event) => event.kind === "node_failed" && event.node_id !== null && !loggedNodeIds.has(event.node_id))
    .map((event) => {
      const traceback = typeof event.detail.traceback === "string" ? event.detail.traceback.trim() : "";
      return {
        nodeId: event.node_id!,
        content: `${event.created_at} ERROR [node] ${event.message}${traceback ? `\n${traceback}` : ""}`,
        truncated: false,
        error: null,
      };
    });
}

export function AllLogsPopover({ onClose }: { onClose: () => void }) {
  const { graph, activeRunId } = useWorkflowStore();
  const [entries, setEntries] = useState<LogEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const nodes = useMemo(
    () => [...graph.nodes].sort((left, right) => left.x - right.x || left.y - right.y || left.id.localeCompare(right.id)),
    [graph.nodes],
  );
  const nodeIndex = useMemo(() => new Map(nodes.map((node, index) => [node.id, index])), [nodes]);

  useEffect(() => {
    if (!activeRunId) {
      setEntries([]);
      return;
    }
    let cancelled = false;
    setLoadError(null);
    setLoading(true);
    Promise.all([
      fetchRun(activeRunId),
      fetchRunErrors(activeRunId).catch(() => []),
      Promise.allSettled(
        nodes.map((node) =>
          fetchNodeLog(activeRunId, node.id)
            .then((log) => ({ nodeId: node.id, content: log.content, truncated: log.truncated, error: log.error })),
        ),
      ),
    ])
      .then(([status, errors, results]) => {
        if (cancelled) return;
        const nodeEntries = results
          .filter((result): result is PromiseFulfilledResult<LogEntry> => result.status === "fulfilled")
          .map((result) => result.value);
        const loggedNodeIds = new Set(nodeEntries.filter((entry) => entry.content || entry.error).map((entry) => entry.nodeId));
        const nodeErrors = nodeFailureEntries(errors, loggedNodeIds);
        const hasNodeFailure = errors.some((event) => event.kind === "node_failed" && event.node_id !== null);
        const runEntry = hasNodeFailure ? null : runFailureEntry(status, errors);
        setEntries([...(runEntry ? [runEntry] : []), ...nodeErrors, ...nodeEntries]);
      })
      .catch((error) => {
        if (!cancelled) {
          setEntries([]);
          setLoadError(String(error?.message ?? error));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeRunId, nodes]);

  const records = useMemo(() => {
    if (!entries) return [];
    let seq = 0;
    const all: LogRecord[] = [];
    for (const entry of entries) {
      const recs = toRecords(entry, seq);
      seq += recs.length;
      all.push(...recs);
    }
    all.sort((left, right) => (left.ts ?? -Infinity) - (right.ts ?? -Infinity) || left.seq - right.seq);
    return all;
  }, [entries]);

  const truncatedNodes = (entries ?? []).filter((entry) => entry.truncated).map((entry) => entry.nodeId);
  const aggregate = records.map((record) => `${record.nodeId}\t${record.text}`).join("\n");

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center bg-gray-900/45 p-6" onClick={onClose}>
      <section
        className="grid max-h-[85vh] w-[min(1040px,calc(100vw-48px))] grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-hidden rounded-lg border border-line bg-panel p-4 shadow-[0_22px_80px_rgba(17,24,39,0.22)]"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3">
          <strong className="text-[13px] text-txt">
            All node logs (merged by time){records.length ? ` · ${records.length} lines` : ""}
          </strong>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              icon="copy"
              disabled={!aggregate}
              onClick={async () => {
                if (!aggregate) throw new Error("merged logs are unavailable");
                await copyText(aggregate);
                showToast("Merged logs copied");
              }}
            >
              Copy all
            </Button>
            <Button variant="ghost" size="sm" icon="x" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
        {loadError ? <p className="font-mono text-[11px] text-red-600">{loadError}</p> : null}
        <div className="overflow-auto rounded-md border border-line bg-panel-2">
          {!activeRunId ? (
            <p className="p-4 font-mono text-[12px] text-txt-mute">Start or select a run to view node logs.</p>
          ) : loading && !entries ? (
            <p className="p-4 font-mono text-[12px] text-txt-mute">Loading logs…</p>
          ) : records.length === 0 ? (
            <p className="p-4 font-mono text-[12px] text-txt-mute">No log lines for any node yet.</p>
          ) : (
            <div className="min-w-max p-3 font-mono text-[11px] leading-relaxed">
              {truncatedNodes.length ? (
                <p className="mb-2 text-[11px] text-amber-700">Showing latest 1 MB for: {truncatedNodes.join(", ")}</p>
              ) : null}
              {records.map((record) => (
                <div key={record.seq} className="flex gap-2 whitespace-pre-wrap break-words">
                  <span
                    className="flex-none select-none font-bold"
                    style={{ color: nodeColor(nodeIndex.get(record.nodeId) ?? 0) }}
                    title={record.nodeId}
                  >
                    {record.nodeId}
                  </span>
                  <span className="min-w-0 flex-1 text-txt-dim">{record.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
