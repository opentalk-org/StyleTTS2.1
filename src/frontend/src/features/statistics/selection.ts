import type { StatisticsSummary } from "./api";


export function reconcileStatisticsEntryId(
  currentId: string | null,
  entries: StatisticsSummary[] | undefined,
): string | null {
  if (entries === undefined) return currentId;
  return currentId !== null && entries.some((entry) => entry.id === currentId) ? currentId : null;
}
