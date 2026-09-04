import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

import { readArtifactJson } from "@/features/artifacts/server";
import { query } from "@/server/clickhouse";
import type { ModelComponent } from "@/shared/types";

interface ArtifactRow {
  runId: string;
  path: string;
}

interface ArrayMetricNameRow {
  name: string;
}

interface ArrayMetricRow {
  step: string;
  timestampMs: string;
  value: number[];
}

const runIdSchema = z.uuid();
const arrayMetricInputSchema = z.object({
  runId: runIdSchema,
  name: z.string().min(1),
});

export const getModelGraph = createServerFn({ method: "GET" })
  .validator(runIdSchema)
  .handler(async ({ data }) => {
    const rows = await query<ArtifactRow>(`
      SELECT toString(run_id) AS runId, path
      FROM artifacts
      WHERE run_id = {run_id:UUID} AND name = 'monitor/model_graph.json'
      ORDER BY timestamp DESC
      LIMIT 1`, { run_id: data });
    const row = rows[0];
    if (row === undefined) throw new Error("Model graph not found");
    return readArtifactJson<ModelComponent[]>(row.runId, row.path);
  });

export const getArrayMetricNames = createServerFn({ method: "GET" })
  .validator(runIdSchema)
  .handler(async ({ data }) => {
    const rows = await query<ArrayMetricNameRow>(`
      SELECT DISTINCT name
      FROM array_metrics
      WHERE run_id = {run_id:UUID}
      ORDER BY name`, { run_id: data });
    return rows.map((row) => row.name);
  });

export const getArrayMetric = createServerFn({ method: "GET" })
  .validator(arrayMetricInputSchema)
  .handler(async ({ data }) => {
    const rows = await query<ArrayMetricRow>(`
      SELECT step, toUnixTimestamp64Milli(max(timestamp)) AS timestampMs,
        argMax(value, timestamp) AS value
      FROM array_metrics
      WHERE run_id = {run_id:UUID} AND name = {name:String}
      GROUP BY step
      ORDER BY step`, { run_id: data.runId, name: data.name });
    return {
      name: data.name,
      steps: rows.map((row) => Number(row.step)),
      timestamps: rows.map((row) => Number(row.timestampMs)),
      values: rows.map((row) => row.value.map(Number)),
    };
  });
