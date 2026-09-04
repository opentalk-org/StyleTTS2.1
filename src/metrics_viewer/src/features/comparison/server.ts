import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { artifactKind } from "@/features/artifacts/server";
import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { Artifact } from "@/shared/types";

interface ArtifactRow {
  runId: string;
  step: string;
  timestamp: string;
  name: string;
  path: string;
  contentType: string;
  sizeBytes: string;
}

interface PlotRow {
  plot: string;
  runId: string;
  x: number;
  y: number;
  wall: number | null;
  rel: number | null;
}

interface DescribeRow {
  name: string;
}

const runIdsSchema = z.array(uuidSchema);
const plotInputSchema = z.object({
  sql: z.string().min(1),
  projectId: uuidSchema,
  runIds: runIdsSchema,
});
const plotRangeInputSchema = z.object({
  runIds: runIdsSchema,
  metric: z.string().min(1),
  xMin: z.number().nullable(),
  xMax: z.number().nullable(),
  targetPoints: z.number().int().min(200).max(4000),
});

interface MetricNameRow {
  name: string;
}

export const getMetricNames = createServerFn({ method: "POST" })
  .validator(runIdsSchema)
  .handler(async ({ data }) => {
    if (data.length === 0) return [];
    const rows = await query<MetricNameRow>(`
      SELECT DISTINCT name
      FROM metrics
      WHERE run_id IN {run_ids:Array(UUID)}
      ORDER BY name`, { run_ids: data });
    return rows.map((row) => row.name);
  });

export const getPlotRange = createServerFn({ method: "POST" })
  .validator(plotRangeInputSchema)
  .handler(async ({ data }) => {
    const rangeFilter = data.xMin === null || data.xMax === null
      ? ""
      : "AND step BETWEEN {x_min:Float64} AND {x_max:Float64}";
    const started = performance.now();
    const rows = await query<PlotRow>(`
      SELECT name AS plot, toString(run_id) AS runId,
        toFloat64(point.1) AS x, toFloat64(point.2) AS y,
        NULL AS wall, NULL AS rel
      FROM (
        SELECT name, run_id,
          largestTriangleThreeBuckets({target_points:UInt32})(step, value) AS points
        FROM metrics
        WHERE run_id IN {run_ids:Array(UUID)}
          AND name = {metric:String}
          ${rangeFilter}
        GROUP BY run_id, name
      )
      ARRAY JOIN points AS point`, {
      run_ids: data.runIds,
      metric: data.metric,
      x_min: data.xMin,
      x_max: data.xMax,
      target_points: data.targetPoints,
    }, { readonly: 2, max_execution_time: 30, max_threads: 4 });
    return {
      plot: rows.map((row) => row.plot),
      runId: rows.map((row) => row.runId),
      x: rows.map((row) => Number(row.x)),
      y: rows.map((row) => Number(row.y)),
      wall: null,
      rel: null,
      elapsedMs: Math.round(performance.now() - started),
    };
  });

export const getArtifacts = createServerFn({ method: "POST" })
  .validator(runIdsSchema)
  .handler(async ({ data }) => {
    const rows = await query<ArtifactRow>(`
      SELECT toString(run_id) AS runId, step,
        toUnixTimestamp64Milli(timestamp) AS timestamp,
        name, path, content_type AS contentType, size_bytes AS sizeBytes
      FROM artifacts
      WHERE run_id IN {run_ids:Array(UUID)}
      ORDER BY name, step`, { run_ids: data });
    return rows.map(toArtifact);
  });

export const runPlotsQuery = createServerFn({ method: "POST" })
  .validator(plotInputSchema)
  .handler(async ({ data }) => {
    const sql = data.sql.trim().replace(/;+$/, "");
    if (!/^(SELECT|WITH)\b/i.test(sql)) throw new Error("Plot query must start with SELECT or WITH");
    const started = performance.now();
    const params = { project_id: data.projectId, run_ids: data.runIds };
    const columns = new Set((await query<DescribeRow>(`DESCRIBE (${sql})`, params, { readonly: 2 })).map((row) => row.name));
    const hasTime = columns.has("wall") && columns.has("rel");
    const rows = await query<PlotRow>(`
      SELECT toString(plot) AS plot, toString(run_id) AS runId,
        toFloat64(x) AS x, toFloat64(y) AS y,
        ${hasTime ? "toFloat64(wall) AS wall, toFloat64(rel) AS rel" : "NULL AS wall, NULL AS rel"}
      FROM (${sql})`, params, {
        readonly: 2,
        max_execution_time: 30,
        max_threads: 4,
        max_result_rows: "2000000",
        optimize_aggregation_in_order: 1,
        result_overflow_mode: "throw",
      });
    const finite = rows.filter((row) => Number.isFinite(row.x) && Number.isFinite(row.y));
    return {
      plot: finite.map((row) => row.plot),
      runId: finite.map((row) => row.runId),
      x: finite.map((row) => Number(row.x)),
      y: finite.map((row) => Number(row.y)),
      wall: hasTime ? finite.map((row) => Number(row.wall)) : null,
      rel: hasTime ? finite.map((row) => Number(row.rel)) : null,
      elapsedMs: Math.round(performance.now() - started),
    };
  });

function toArtifact(row: ArtifactRow): Artifact {
  return {
    id: `${row.runId}-${row.name}-${row.step}`,
    runId: row.runId,
    name: row.name,
    step: Number(row.step),
    timestamp: Number(row.timestamp),
    kind: artifactKind(row.contentType, row.name),
    contentType: row.contentType,
    sizeBytes: Number(row.sizeBytes),
    source: `/api/artifacts/content?run_id=${encodeURIComponent(row.runId)}&path=${encodeURIComponent(row.path)}`,
  };
}
