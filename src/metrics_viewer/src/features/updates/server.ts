import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { artifactKind } from "@/features/artifacts/server";
import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { Artifact, RunStatus } from "@/shared/types";

const epoch = "1970-01-01 00:00:00.000000000";
const cursorSchema = z.object({
  status: z.string(),
  metrics: z.string(),
  arrayMetrics: z.string(),
  artifacts: z.string(),
});
const changesInputSchema = z.object({
  projectId: uuidSchema,
  runIds: z.array(uuidSchema),
  watchMetrics: z.boolean(),
  watchArrayMetrics: z.boolean(),
  watchArtifacts: z.boolean(),
  cursor: cursorSchema,
});

interface RunChangeRow {
  id: string;
  projectId: string;
  name: string;
  status: RunStatus;
  startedAt: string;
  endedAt: string;
  cursor: string;
}

interface MetricChangeRow {
  cursor: string;
  runId: string;
  name: string;
  value: number;
  minStep: number;
  maxStep: number;
}

interface ArrayMetricChangeRow {
  cursor: string;
  runId: string;
  name: string;
  step: string;
  timestampMs: string;
  value: number[];
}

interface ArtifactChangeRow {
  cursor: string;
  runId: string;
  step: string;
  timestamp: string;
  name: string;
  path: string;
  contentType: string;
  sizeBytes: string;
}

export interface UpdateCursor {
  status: string;
  metrics: string;
  arrayMetrics: string;
  artifacts: string;
}

export const initialUpdateCursor: UpdateCursor = {
  status: epoch,
  metrics: epoch,
  arrayMetrics: epoch,
  artifacts: epoch,
};

export const pollVisibleChanges = createServerFn({ method: "POST" })
  .validator(changesInputSchema)
  .handler(async ({ data }) => {
    const [status, metrics, arrayMetrics, artifacts] = await Promise.all([
      statusChanges(data.projectId, data.cursor.status),
      data.watchMetrics ? scalarMetricChanges(data.runIds, data.cursor.metrics) : emptyMetricChange(data.cursor.metrics),
      data.watchArrayMetrics
        ? arrayMetricChanges(data.runIds, data.cursor.arrayMetrics)
        : emptyArrayMetricChange(data.cursor.arrayMetrics),
      data.watchArtifacts ? artifactChanges(data.runIds, data.cursor.artifacts) : emptyArtifactChange(data.cursor.artifacts),
    ]);
    return {
      cursor: {
        status: status.cursor,
        metrics: metrics.cursor,
        arrayMetrics: arrayMetrics.cursor,
        artifacts: artifacts.cursor,
      },
      runs: status.runs,
      metrics: metrics.rows,
      arrayMetrics: arrayMetrics.rows,
      artifacts: artifacts.rows,
    };
  });

async function statusChanges(projectId: string, after: string) {
  if (after === epoch) {
    return {
      cursor: await latestTimestamp("run_status", "run_id IN (SELECT id FROM runs WHERE project_id = {project_id:UUID})", { project_id: projectId }),
      runs: [],
    };
  }
  const rows = await query<RunChangeRow>(`
    SELECT toString(r.id) AS id, toString(r.project_id) AS projectId, r.name,
      toString(s.status) AS status,
      toUnixTimestamp64Milli(s.started_at) AS startedAt,
      if(s.status IN ('succeeded', 'failed', 'cancelled'),
        toUnixTimestamp64Milli(s.last_status_at), 0) AS endedAt,
      toString(changes.cursor) AS cursor
    FROM runs AS r
    INNER JOIN (
      SELECT run_id, min(timestamp) AS started_at, max(timestamp) AS last_status_at,
        argMax(status, timestamp) AS status
      FROM run_status
      GROUP BY run_id
    ) AS s ON s.run_id = r.id
    CROSS JOIN (
      SELECT max(timestamp) AS cursor
      FROM run_status
      WHERE run_id IN (SELECT id FROM runs WHERE project_id = {project_id:UUID})
        AND timestamp > {after:DateTime64(9)}
    ) AS changes
    WHERE r.project_id = {project_id:UUID}
      AND r.id IN (
        SELECT run_id FROM run_status
        WHERE timestamp > {after:DateTime64(9)}
      )`, { project_id: projectId, after });
  return {
    cursor: rows[0]?.cursor ?? after,
    runs: rows.map((row) => ({
      id: row.id,
      projectId: row.projectId,
      name: row.name,
      status: row.status,
      startedAt: Number(row.startedAt),
      endedAt: Number(row.endedAt),
      params: {},
      summary: {},
    })),
  };
}

async function scalarMetricChanges(runIds: string[], after: string) {
  if (runIds.length === 0) return emptyMetricChange(after);
  if (after === epoch) {
    return { cursor: await latestTimestamp("metrics", "run_id IN {run_ids:Array(UUID)}", { run_ids: runIds }), rows: [] };
  }
  const rows = await query<MetricChangeRow>(`
    WITH (SELECT max(timestamp) FROM metrics
      WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}) AS next_cursor
    SELECT toString(next_cursor) AS cursor,
      toString(run_id) AS runId, name, toFloat64(argMax(value, step)) AS value,
      toFloat64(min(step)) AS minStep, toFloat64(max(step)) AS maxStep
    FROM metrics
    WHERE run_id IN {run_ids:Array(UUID)}
      AND timestamp > {after:DateTime64(9)}
    GROUP BY run_id, name`, { run_ids: runIds, after });
  return { cursor: rows[0]?.cursor ?? after, rows };
}

async function arrayMetricChanges(runIds: string[], after: string) {
  if (runIds.length === 0) return emptyArrayMetricChange(after);
  if (after === epoch) {
    return { cursor: await latestTimestamp("array_metrics", "run_id IN {run_ids:Array(UUID)}", { run_ids: runIds }), rows: [] };
  }
  const rows = await query<ArrayMetricChangeRow>(`
    WITH (SELECT max(timestamp) FROM array_metrics
      WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}) AS next_cursor
    SELECT toString(next_cursor) AS cursor,
      toString(run_id) AS runId, name, step,
      toUnixTimestamp64Milli(max(timestamp)) AS timestampMs,
      argMax(value, timestamp) AS value
    FROM array_metrics
    WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}
    GROUP BY run_id, name, step`, { run_ids: runIds, after });
  return { cursor: rows[0]?.cursor ?? after, rows };
}

async function artifactChanges(runIds: string[], after: string) {
  if (runIds.length === 0) return emptyArtifactChange(after);
  if (after === epoch) {
    return { cursor: await latestTimestamp("artifacts", "run_id IN {run_ids:Array(UUID)}", { run_ids: runIds }), rows: [] };
  }
  const rows = await query<ArtifactChangeRow>(`
    WITH (SELECT max(timestamp) FROM artifacts
      WHERE run_id IN {run_ids:Array(UUID)} AND timestamp > {after:DateTime64(9)}) AS next_cursor
    SELECT toString(next_cursor) AS cursor,
      toString(run_id) AS runId, step,
      toUnixTimestamp64Milli(timestamp) AS timestamp,
      name, path, content_type AS contentType, size_bytes AS sizeBytes
    FROM artifacts
    WHERE run_id IN {run_ids:Array(UUID)}
      AND timestamp > {after:DateTime64(9)}`, { run_ids: runIds, after });
  return { cursor: rows[0]?.cursor ?? after, rows: rows.map(toArtifact) };
}

function emptyMetricChange(cursor: string) {
  return { cursor, rows: [] as MetricChangeRow[] };
}

function emptyArrayMetricChange(cursor: string) {
  return { cursor, rows: [] as ArrayMetricChangeRow[] };
}

function emptyArtifactChange(cursor: string) {
  return { cursor, rows: [] as Artifact[] };
}

async function latestTimestamp(
  table: "run_status" | "metrics" | "array_metrics" | "artifacts",
  filter: string,
  params: Record<string, unknown>,
) {
  const rows = await query<{ cursor: string }>(`
    SELECT toString(max(timestamp)) AS cursor
    FROM ${table}
    WHERE ${filter}`, params);
  return rows[0]?.cursor ?? epoch;
}

function toArtifact(row: ArtifactChangeRow): Artifact {
  const kind = artifactKind(row.contentType, row.name);
  return {
    id: `${row.runId}-${row.name}-${row.step}`,
    runId: row.runId,
    name: row.name,
    step: Number(row.step),
    timestamp: Number(row.timestamp),
    kind,
    contentType: row.contentType,
    sizeBytes: Number(row.sizeBytes),
    source: kind === "plot" || kind === "text"
      ? row.path
      : `/api/artifacts/content?run_id=${encodeURIComponent(row.runId)}&path=${encodeURIComponent(row.path)}`,
  };
}
