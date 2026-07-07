import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { deleteStatisticsEntry, fetchStatisticsEntries, fetchStatisticsEntry } from "./api";

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
