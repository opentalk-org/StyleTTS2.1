import { createServerFn } from "@tanstack/react-start";

import { query } from "@/server/clickhouse";

interface ProjectRow {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  lastRunAt: string;
  runCount: string;
  runningCount: string;
}

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

export const listProjects = createServerFn({ method: "GET" })
  .handler(async () => {
    const rows = await query<ProjectRow>(projectsSql);
    return rows.map((row) => ({
      ...row,
      createdAt: Number(row.createdAt),
      lastRunAt: Number(row.lastRunAt),
      runCount: Number(row.runCount),
      runningCount: Number(row.runningCount),
    }));
  });
