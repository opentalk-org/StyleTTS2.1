import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import type { WorkflowSchema } from "../workflows/types";
import { deleteStatisticsEntry, fetchStatisticsEntries, fetchStatisticsEntry } from "./api";
import { computeDatasetStatistics } from "./workflow";

export const STATISTICS_KEY = "statistics";

export function useStatisticsEntriesQuery() {
  return useQuery({
    queryKey: [STATISTICS_KEY],
    queryFn: fetchStatisticsEntries,
  });
}

export function useStatisticsEntryQuery(id: string | null) {
  return useQuery({
    queryKey: [STATISTICS_KEY, id],
    queryFn: () => fetchStatisticsEntry(id as string),
    enabled: id != null,
  });
}

export function useStatisticsActions() {
  const qc = useQueryClient();

  const remove = useMutation({
    mutationFn: deleteStatisticsEntry,
    onSuccess: () => {
      showToast("Statistics entry deleted", undefined, "error");
      qc.invalidateQueries({ queryKey: [STATISTICS_KEY] });
    },
  });

  return {
    remove: (id: string) => remove.mutateAsync(id),
  };
}

export function useComputeStatisticsMutation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ schema, datasetId, name }: { schema: WorkflowSchema; datasetId: string; name: string }) =>
      computeDatasetStatistics(schema, datasetId, name),
    onSuccess: () => {
      showToast("Statistics computed");
      qc.invalidateQueries({ queryKey: [STATISTICS_KEY] });
    },
    onError: (error) => showToast("Could not compute statistics", String(error), "error"),
  });
}
