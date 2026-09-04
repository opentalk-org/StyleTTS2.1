import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";

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

const projectsSql = (filter = "") => `
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
${filter}
ORDER BY lastRunAt DESC, p.name ASC`;

export const listProjects = createServerFn({ method: "GET" })
  .handler(async () => {
    return projectRows(await query<ProjectRow>(projectsSql()));
  });

interface ProjectChangeRow {
  cursor: string;
  ids: string[];
}

export const pollProjectChanges = createServerFn({ method: "POST" })
  .validator(z.string())
  .handler(async ({ data }) => {
    const changes = await query<ProjectChangeRow>(`
      SELECT toString(max(timestamp)) AS cursor,
        groupUniqArray(toString(project_id)) AS ids
      FROM (
        SELECT id AS project_id, updated_at AS timestamp
        FROM projects FINAL
        WHERE updated_at > {after:DateTime64(9)}
        UNION ALL
        SELECT r.project_id, s.timestamp
        FROM run_status AS s
        INNER JOIN runs AS r ON r.id = s.run_id
        WHERE s.timestamp > {after:DateTime64(9)}
      )`, { after: data });
    const change = changes[0];
    if (change === undefined || change.ids.length === 0) return { cursor: data, projects: [] };
    if (data === "1970-01-01 00:00:00.000000000") return { cursor: change.cursor, projects: [] };
    const rows = await query<ProjectRow>(projectsSql("WHERE p.id IN {project_ids:Array(UUID)}"), {
      project_ids: change.ids,
    });
    return { cursor: change.cursor, projects: projectRows(rows) };
  });

function projectRows(rows: ProjectRow[]) {
  return rows.map((row) => ({
    ...row,
    createdAt: Number(row.createdAt),
    lastRunAt: Number(row.lastRunAt),
    runCount: Number(row.runCount),
    runningCount: Number(row.runningCount),
  }));
}
