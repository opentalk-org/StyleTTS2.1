import { createServerFn } from "@tanstack/react-start";

import { query } from "@/server/clickhouse";
import { uuidSchema } from "@/shared/ids";
import type { Checkpoint, Lineage, LineageRun, RunStatus } from "@/shared/types";

interface CheckpointRow {
  id: string;
  runId: string;
  ancestorId: string;
  name: string;
  step: string;
  createdAt: string;
  sizeBytes: string;
  type: string;
  runName: string;
  runStatus: RunStatus;
  firstStep: string;
}

const NIL_UUID = "00000000-0000-0000-0000-000000000000";

/**
 * Checkpoints of a project, each with the checkpoint it was resumed from.
 *
 * `assets.run_id` and `assets.ancestor_asset_id` are the two columns the graph is built
 * from. A checkpoint with no run cannot be placed in a lane, so it is left out; an
 * ancestor outside the project simply reads as a root, which the layout already handles.
 *
 * The checkpoint step is normalized when the asset is written or migrated.
 */
const checkpointsSql = `
WITH
current_status AS (
  SELECT run_id, argMax(status, timestamp) AS status
  FROM run_status
  GROUP BY run_id
),
metric_starts AS (
  SELECT run_id, min(step) AS first_step
  FROM metrics
  WHERE run_id IN (SELECT id FROM runs WHERE project_id = {project_id:UUID})
    AND name NOT LIKE 'system/%'
  GROUP BY run_id
)
SELECT
  toString(a.id) AS id,
  toString(a.run_id) AS runId,
  if(a.ancestor_asset_id = toUUID({nil:String}), '', toString(a.ancestor_asset_id)) AS ancestorId,
  a.name AS name,
  toString(a.step) AS step,
  toString(toUnixTimestamp64Milli(a.updated_at)) AS createdAt,
  toString(a.size) AS sizeBytes,
  a.type AS type,
  r.name AS runName,
  toString(s.status) AS runStatus,
  toString(ifNull(m.first_step, 0)) AS firstStep
FROM assets AS a FINAL
INNER JOIN runs AS r ON r.id = a.run_id
INNER JOIN current_status AS s ON s.run_id = a.run_id
LEFT JOIN metric_starts AS m ON m.run_id = a.run_id
WHERE a.kind = 'checkpoint'
  AND a.run_id != toUUID({nil:String})
  AND r.project_id = {project_id:UUID}
ORDER BY a.updated_at ASC`;

export const getLineage = createServerFn({ method: "GET" })
  .validator(uuidSchema)
  .handler(async ({ data }): Promise<Lineage> => {
    const rows = await query<CheckpointRow>(checkpointsSql, { project_id: data, nil: NIL_UUID });
    if (rows.length === 0) return { checkpoints: [], runs: [], firstSteps: {} };

    const checkpoints: Checkpoint[] = rows.map((row) => ({
      id: row.id,
      runId: row.runId,
      ancestorId: row.ancestorId === "" ? null : row.ancestorId,
      name: row.name,
      step: Number(row.step),
      createdAt: Number(row.createdAt),
      sizeBytes: Number(row.sizeBytes),
      type: row.type,
    }));
    const byRun = new Map<string, LineageRun>();
    const firstSteps: Record<string, number> = {};
    for (const row of rows) {
      byRun.set(row.runId, { id: row.runId, name: row.runName, status: row.runStatus });
      firstSteps[row.runId] = Number(row.firstStep);
    }
    const runs = [...byRun.values()];
    return { checkpoints, runs, firstSteps };
  });
