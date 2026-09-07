import { useInfiniteQuery } from "@tanstack/react-query";

import { getLogs, type LogCursor } from "./server";

export function useLogsQuery(runIds: string[], enabled: boolean) {
  const ids = [...runIds].sort();
  return useInfiniteQuery({
    queryKey: ["logs", ids],
    queryFn: ({ pageParam }) => getLogs({ data: { runIds: ids, before: pageParam } }),
    initialPageParam: null as LogCursor | null,
    getNextPageParam: (page) => page.nextCursor ?? undefined,
    enabled: enabled && ids.length > 0,
    staleTime: Infinity,
  });
}
