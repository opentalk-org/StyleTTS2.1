import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { showToast } from "@/shared/feedback/Toast";
import { type JobQuery, deleteAllJobs, deleteJob, deleteJobs, fetchJobs, renameJob, stopJob } from "./api";

const JOBS_KEY = "jobs";

export function useJobsQuery(params: JobQuery) {
  return useQuery({
    queryKey: [JOBS_KEY, params],
    queryFn: () => fetchJobs(params),
    placeholderData: keepPreviousData,
  });
}

export function useJobActions() {
  const queryClient = useQueryClient();
  const invalidateJobs = () => queryClient.invalidateQueries({ queryKey: [JOBS_KEY] });

  const stop = useMutation({
    mutationFn: stopJob,
    onSuccess: (run) => {
      showToast(`Stop requested for ${run.run_id}`);
      invalidateJobs();
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
  const remove = useMutation({
    mutationFn: deleteJob,
    onSuccess: () => {
      showToast("Job removed", undefined, "error");
      invalidateJobs();
    },
  });
  const rename = useMutation({
    mutationFn: ({ runId, name }: { runId: string; name: string }) => renameJob(runId, name),
    onSuccess: () => {
      showToast("Job renamed");
      invalidateJobs();
    },
  });
  const removeMany = useMutation({
    mutationFn: deleteJobs,
    onSuccess: (_data, runIds) => {
      showToast(`Removed ${runIds.length} job${runIds.length === 1 ? "" : "s"}`, undefined, "error");
      invalidateJobs();
    },
  });
  const removeAll = useMutation({
    mutationFn: deleteAllJobs,
    onSuccess: () => {
      showToast("Removed all jobs", undefined, "error");
      invalidateJobs();
    },
  });

  return {
    remove: (runId: string) => remove.mutate(runId),
    stop: (runId: string) => stop.mutate(runId),
    rename: (runId: string, name: string) => rename.mutate({ runId, name }),
    removeMany: (runIds: string[], options?: { onSuccess?: () => void }) => removeMany.mutate(runIds, options),
    removeAll: (options?: { onSuccess?: () => void }) => removeAll.mutate(undefined, options),
    removing: remove.isPending,
    stopping: stop.isPending,
    renaming: rename.isPending,
    removingMany: removeMany.isPending,
    removingAll: removeAll.isPending,
  };
}
