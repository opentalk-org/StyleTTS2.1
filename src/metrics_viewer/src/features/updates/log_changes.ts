import { query } from "@/server/clickhouse";

export interface LogChangeRow {
  cursor: string;
  runId: string;
  timestampMs: string;
  message: string;
}

export async function logChanges(runIds: string[], after: string, epoch: string, emptyBaseline: string) {
  if (runIds.length === 0) return { cursor: after, rows: [] as LogChangeRow[] };
  if (after === epoch) {
    const latest = await query<{ cursor: string }>(`
      SELECT toString(max(timestamp)) AS cursor
      FROM logs
      WHERE run_id IN {run_ids:Array(UUID)}`, { run_ids: runIds });
    const cursor = latest[0]?.cursor ?? epoch;
    return { cursor: cursor === epoch ? emptyBaseline : cursor, rows: [] as LogChangeRow[] };
  }
  const rows = await query<LogChangeRow>(`
    WITH (SELECT max(timestamp) FROM logs
      WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}) AS next_cursor
    SELECT toString(next_cursor) AS cursor, toString(run_id) AS runId,
      toUnixTimestamp64Milli(timestamp) AS timestampMs, message
    FROM logs
    WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}
    ORDER BY timestamp DESC, run_id DESC, message DESC`, { run_ids: runIds, after });
  return { cursor: rows[0]?.cursor ?? after, rows };
}
