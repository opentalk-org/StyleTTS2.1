import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { artifactKind } from "@/features/artifacts/server";
import { query } from "@/server/clickhouse";
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
}

const runIdsSchema = z.array(z.uuid());
const plotInputSchema = z.object({
  sql: z.string().min(1),
  projectId: z.uuid(),
  runIds: runIdsSchema,
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
    const rows = await query<PlotRow>(`
      SELECT toString(plot) AS plot, toString(run_id) AS runId,
        toFloat64(x) AS x, toFloat64(y) AS y
      FROM (${sql})`, { project_id: data.projectId, run_ids: data.runIds }, {
        readonly: 2,
        max_execution_time: 30,
        max_result_rows: "2000000",
        result_overflow_mode: "throw",
      });
    const finite = rows.filter((row) => Number.isFinite(row.x) && Number.isFinite(row.y));
    return {
      plot: finite.map((row) => row.plot),
      runId: finite.map((row) => row.runId),
      x: finite.map((row) => Number(row.x)),
      y: finite.map((row) => Number(row.y)),
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
