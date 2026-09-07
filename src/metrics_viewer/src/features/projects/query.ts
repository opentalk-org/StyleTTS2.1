import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import type { Project } from "@/shared/types";

import { listProjects, pollProjectChanges } from "./server";

export function useProjectsQuery() {
  const queryClient = useQueryClient();
  const cursor = useRef("1970-01-01 00:00:00.000000000");
  const projects = useQuery({ queryKey: ["projects"], queryFn: () => listProjects(), staleTime: Infinity });
  const changes = useQuery({
    queryKey: ["project-updates"],
    queryFn: () => pollProjectChanges({ data: cursor.current }),
    refetchInterval: 3_000,
    refetchIntervalInBackground: true,
    staleTime: Infinity,
  });
  useEffect(() => {
    const update = changes.data;
    if (update === undefined) return;
    cursor.current = update.cursor;
    if (update.projects.length === 0) return;
    queryClient.setQueryData<Project[]>(["projects"], (current) => {
      const changed = new Map(update.projects.map((project) => [project.id, project]));
      const merged = (current ?? []).map((project) => changed.get(project.id) ?? project);
      const known = new Set(merged.map((project) => project.id));
      return [...update.projects.filter((project) => !known.has(project.id)), ...merged]
        .sort((a, b) => b.lastRunAt - a.lastRunAt || a.name.localeCompare(b.name));
    });
  }, [changes.data, queryClient]);
  return projects;
}
