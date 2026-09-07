import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { JsonValue, ProjectColumns, Run, RunStatus, Scalar } from "@/shared/types";

interface RunRow {
  id: string;
  projectId: string;
  name: string;
  status: RunStatus;
  startedAt: string;
  endedAt: string;
  paramNames: string[];
  metricNames: string[];
  trainingConfig: string;
}

interface RunDetailRow {
  runId: string;
  trainingConfig: string;
  summaryJson: string;
}

interface RunMetricRow {
  runId: string;
  name: string;
  value: number;
}

const runIdsSchema = uuidSchema.array();
const runIdSchema = uuidSchema;
const runMetricsSchema = z.object({ projectId: uuidSchema, names: z.array(z.string()) });

export const getProjectBootstrap = createServerFn({ method: "GET" })
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
        ) AS endedAt,
        JSONExtractKeys(r.train_config) AS paramNames,
        (SELECT groupUniqArray(name) FROM metrics
          WHERE run_id IN (SELECT id FROM runs WHERE project_id = {project_id:UUID})) AS metricNames,
        r.train_config AS trainingConfig
      FROM runs AS r
      INNER JOIN (
        SELECT run_id, min(timestamp) AS started_at, max(timestamp) AS last_status_at,
          argMax(status, timestamp) AS status
        FROM run_status
        GROUP BY run_id
      ) AS s ON s.run_id = r.id
      WHERE r.project_id = {project_id:UUID}
      ORDER BY s.started_at DESC`, { project_id: data });
    const runs: Run[] = rows.map((row) => ({
      id: row.id,
      projectId: row.projectId,
      name: row.name,
      status: row.status,
      startedAt: Number(row.startedAt),
      endedAt: Number(row.endedAt),
      params: {},
      summary: {},
    }));
    const columns: ProjectColumns = {
      params: [...new Set(rows.flatMap((row) => row.paramNames))].sort(),
      metrics: [...new Set(rows.flatMap((row) => row.metricNames))].sort(),
    };
    return { runs, columns };
  });

export const getRunMetrics = createServerFn({ method: "POST" })
  .validator(runMetricsSchema)
  .handler(async ({ data }) => {
    const rows = await query<RunMetricRow>(`
      SELECT toString(m.run_id) AS runId, m.name,
        toFloat64(argMax(m.value, m.step)) AS value
      FROM metrics AS m
      INNER JOIN runs AS r ON r.id = m.run_id
      WHERE r.project_id = {project_id:UUID}
        AND m.name IN {names:Array(String)}
      GROUP BY m.run_id, m.name
    `, { project_id: data.projectId, names: data.names }, {
      optimize_aggregation_in_order: 1,
    });
    const summaries: Record<string, Record<string, number>> = {};
    for (const row of rows) {
      const summary = summaries[row.runId] ?? {};
      summary[row.name] = Number(row.value);
      summaries[row.runId] = summary;
    }
    return summaries;
  });

export const getRunDetails = createServerFn({ method: "POST" })
  .validator(runIdsSchema)
  .handler(async ({ data }) => {
    const rows = await query<RunDetailRow>(`
    SELECT toString(r.id) AS runId, r.train_config AS trainingConfig,
      if(empty(s.summary_json), '{}', s.summary_json) AS summaryJson
    FROM runs AS r
    LEFT JOIN (
      SELECT run_id,
        toJSONString(mapFromArrays(groupArray(name), groupArray(value))) AS summary_json
      FROM (
        SELECT run_id, name, toFloat64(argMax(value, step)) AS value
        FROM metrics
        WHERE run_id IN {run_ids:Array(UUID)}
        GROUP BY run_id, name
      )
      GROUP BY run_id
    ) AS s ON s.run_id = r.id
    WHERE r.id IN {run_ids:Array(UUID)}
    `, { run_ids: data }, {
      optimize_aggregation_in_order: 1,
    });
    const details: Record<string, { params: Record<string, Scalar>; summary: Record<string, number> }> = {};
    for (const runId of data) details[runId] = { params: {}, summary: {} };
    for (const row of rows) {
      details[row.runId] = {
        params: scalarParams(JSON.parse(row.trainingConfig) as Record<string, unknown>),
        summary: JSON.parse(row.summaryJson) as Record<string, number>,
      };
    }
    return details;
  });

export const getRunConfig = createServerFn({ method: "GET" })
  .validator(runIdSchema)
  .handler(async ({ data }) => {
    const rows = await query<{ dataConfig: string; trainConfig: string }>(`
      SELECT data_config AS dataConfig, train_config AS trainConfig
      FROM runs
      WHERE id = {run_id:UUID}
      LIMIT 1`, { run_id: data });
    const row = rows[0];
    if (row === undefined) throw new Error(`Run not found: ${data}`);
    return {
      data: JSON.parse(row.dataConfig) as JsonValue,
      training: JSON.parse(row.trainConfig) as JsonValue,
    };
  });

function scalarParams(config: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(config).filter((entry): entry is [string, Scalar] =>
      typeof entry[1] === "string" || typeof entry[1] === "number" || typeof entry[1] === "boolean"
    ),
  );
}
