import { useQuery } from "@tanstack/react-query";

import { listProjects } from "./server";

export function useProjectsQuery() {
  return useQuery({ queryKey: ["projects"], queryFn: () => listProjects() });
}
