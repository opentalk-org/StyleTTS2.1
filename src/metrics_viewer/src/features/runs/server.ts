import { createServerFn } from "@tanstack/react-start";

import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { RunStatus, Scalar } from "@/shared/types";

interface RunRow {
  id: string;
  projectId: string;
  name: string;
  status: RunStatus;
  startedAt: string;
  endedAt: string;
}

interface RunConfigRow {
  id: string;
  trainingConfig: string;
}

interface SummaryRow {
  runId: string;
  name: string;
  value: number;
}

export const listRuns = createServerFn({ method: "GET" })
  .validator(uuidSchema)
  .handler(async ({ data }) => {
    const rows = await query<RunRow>(`
      SELECT toString(r.id) AS id, toString(r.project_id) AS projectId, r.name,
        toString(s.status) AS status,
        toUnixTimestamp64Milli(s.started_at) AS startedAt,
        if(
          s.status IN ('succeeded', 'failed', 'cancelled'),
          toUnixTimestamp64Milli(s.last_status_at),
          0
        ) AS endedAt
      FROM runs AS r
      INNER JOIN (
        SELECT run_id, min(timestamp) AS started_at, max(timestamp) AS last_status_at,
          argMax(status, timestamp) AS status
        FROM run_status
        GROUP BY run_id
      ) AS s ON s.run_id = r.id
      WHERE r.project_id = {project_id:UUID}
      ORDER BY s.started_at DESC`, { project_id: data });
    return rows.map((row) => ({
      ...row,
      startedAt: Number(row.startedAt),
      endedAt: Number(row.endedAt),
      params: {},
      summary: {},
    }));
  });

export const getRunParams = createServerFn({ method: "GET" })
  .validator(uuidSchema)
  .handler(async ({ data }) => {
    const rows = await query<RunConfigRow>(`
      SELECT toString(id) AS id, train_config AS trainingConfig
      FROM runs
      WHERE id = {run_id:UUID}`, { run_id: data });
    const row = rows[0];
    if (row === undefined) throw new Error(`Run ${data} not found`);
    return scalarParams(JSON.parse(row.trainingConfig) as Record<string, unknown>);
  });

export const getRunSummary = createServerFn({ method: "GET" })
  .validator(uuidSchema)
  .handler(async ({ data }) => {
    const summaries = await runSummaries([data]);
    return summaries.get(data) ?? {};
  });

async function runSummaries(runIds: string[]) {
  const summaries = new Map<string, Record<string, number>>();
  if (runIds.length === 0) return summaries;
  const rows = await query<SummaryRow>(`
    SELECT toString(run_id) AS runId, name, toFloat64(argMax(value, step)) AS value
    FROM metrics
    WHERE run_id IN {run_ids:Array(UUID)}
    GROUP BY run_id, name`, { run_ids: runIds }, {
      optimize_aggregation_in_order: 1,
    });
  for (const row of rows) {
    const values = summaries.get(row.runId) ?? {};
    values[row.name] = Number(row.value);
    summaries.set(row.runId, values);
  }
  return summaries;
}

function scalarParams(config: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(config).filter((entry): entry is [string, Scalar] =>
      typeof entry[1] === "string" || typeof entry[1] === "number" || typeof entry[1] === "boolean"
    ),
  );
}
