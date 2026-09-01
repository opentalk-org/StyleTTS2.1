import * as mock from "@/data/mock";
import { runPlotQuery } from "@/data/plotQuery";
import { cached } from "@/shared/cache";
import type { PlotQueryResult } from "@/shared/types";

export const { getArtifacts, listProjects, listRuns } = mock;

/**
 * Runs the query that defines the plots. A real deployment posts the SQL to
 * ClickHouse — which is what the cache is for; here it is evaluated locally
 * against the generated series.
 */
export async function runPlotsQuery(
  sql: string,
  projectId: string,
  selectedRunIds: string[],
): Promise<PlotQueryResult> {
  const key = `plots:${projectId}:${[...selectedRunIds].sort().join(",")}:${hash(sql)}`;
  return cached(key, async () =>
    runPlotQuery(sql, {
      selectedRunIds,
      projectRunIds: mock.runIdsForProject(projectId),
      loadPoints: mock.pointsFor,
    }),
  );
}

function hash(value: string): number {
  return Array.from(value).reduce((total, char) => ((total << 5) - total + char.charCodeAt(0)) | 0, 0);
}
