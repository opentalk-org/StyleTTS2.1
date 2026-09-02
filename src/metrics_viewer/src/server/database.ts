import { createClient, type ClickHouseClient } from "@clickhouse/client";

import type {
  ArtifactRow,
  PlotRow,
  ProjectRow,
  RunRow,
  SummaryRow,
} from "./types.js";
import { params } from "./types.js";

const projectsSql = `
SELECT
  toString(p.id) AS id,
  p.name AS name,
  p.description AS description,
  toUnixTimestamp64Milli(p.created_at) AS createdAt,
  ifNull(r.last_run_at, 0) AS lastRunAt,
  ifNull(r.run_count, 0) AS runCount,
  ifNull(r.running_count, 0) AS runningCount
FROM projects AS p FINAL
LEFT JOIN (
  SELECT
    r.project_id,
    max(toUnixTimestamp64Milli(s.started_at)) AS last_run_at,
    count() AS run_count,
    countIf(s.status = 'running') AS running_count
  FROM runs AS r
  INNER JOIN (
    SELECT
      run_id,
      min(timestamp) AS started_at,
      argMax(status, timestamp) AS status
    FROM run_status
    GROUP BY run_id
  ) AS s ON s.run_id = r.id
  GROUP BY r.project_id
) AS r ON r.project_id = p.id
ORDER BY lastRunAt DESC, p.name ASC`;

export interface DatabaseConfig {
  url: string;
  username: string;
  password: string;
}

export class Database {
  readonly client: ClickHouseClient;

  constructor(config: DatabaseConfig) {
    this.client = createClient(config);
  }

  async projects() {
    const rows = await this.query<ProjectRow>(projectsSql);
    return rows.map((row) => ({
      ...row,
      createdAt: Number(row.createdAt),
      lastRunAt: Number(row.lastRunAt),
      runCount: Number(row.runCount),
      runningCount: Number(row.runningCount),
    }));
  }

  async runs(projectId: string) {
    const rows = await this.query<RunRow>(`
      SELECT toString(r.id) AS id, toString(r.project_id) AS projectId, r.name,
        toString(s.status) AS status,
        toUnixTimestamp64Milli(s.started_at) AS startedAt,
        if(
          s.status IN ('succeeded', 'failed', 'cancelled'),
          toUnixTimestamp64Milli(s.last_status_at),
          0
        ) AS endedAt,
        r.train_config AS trainingConfig
      FROM runs AS r
      INNER JOIN (
        SELECT
          run_id,
          min(timestamp) AS started_at,
          max(timestamp) AS last_status_at,
          argMax(status, timestamp) AS status
        FROM run_status
        GROUP BY run_id
      ) AS s ON s.run_id = r.id
      WHERE r.project_id = {project_id:UUID}
      ORDER BY s.started_at DESC`, { project_id: projectId });
    const ids = rows.map((row) => row.id);
    const summaries = new Map<string, Record<string, number>>();
    if (ids.length > 0) {
      const summaryRows = await this.query<SummaryRow>(`
        SELECT toString(run_id) AS runId, name, toFloat64(argMax(value, step)) AS value
        FROM metrics
        WHERE run_id IN {run_ids:Array(UUID)}
        GROUP BY run_id, name`, { run_ids: ids });
      for (const row of summaryRows) {
        const values = summaries.get(row.runId) ?? {};
        values[row.name] = Number(row.value);
        summaries.set(row.runId, values);
      }
    }
    return rows.map(({ trainingConfig, ...row }) => ({
      ...row,
      startedAt: Number(row.startedAt),
      endedAt: Number(row.endedAt),
      params: params(trainingConfig),
      summary: summaries.get(row.id) ?? {},
    }));
  }

  async artifacts(runIds: string[]) {
    if (runIds.length === 0) return [];
    return this.query<ArtifactRow>(`
      SELECT toString(run_id) AS runId, step,
        toUnixTimestamp64Milli(timestamp) AS timestamp,
        name, path, content_type AS contentType, size_bytes AS sizeBytes
      FROM artifacts
      WHERE run_id IN {run_ids:Array(UUID)}
      ORDER BY name, step`, { run_ids: runIds });
  }

  async plots(sql: string, projectId: string, runIds: string[]) {
    const started = performance.now();
    const rows = await this.query<PlotRow>(`
      SELECT toString(plot) AS plot, toString(run_id) AS runId,
        toFloat64(x) AS x, toFloat64(y) AS y
      FROM (${sql})`, { project_id: projectId, run_ids: runIds }, {
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
  }

  private async query<T>(
    query: string,
    query_params: Record<string, unknown> = {},
    clickhouse_settings: Record<string, string | number> = {},
  ): Promise<T[]> {
    const result = await this.client.query({
      query,
      query_params,
      clickhouse_settings,
      format: "JSONEachRow",
    });
    return result.json<T>();
  }
}
