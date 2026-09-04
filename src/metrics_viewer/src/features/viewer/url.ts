import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

import type { PanelTab } from "@/shared/types";

import type { ViewerSearch } from "./search";
import { useViewerStore } from "./store";

function toSearch(projectId: string | null, runIds: string[], tab: PanelTab): ViewerSearch {
  return {
    project: projectId ?? undefined,
    runs: runIds.length === 0 ? undefined : runIds.join(","),
    tab: projectId === null || tab === "charts" ? undefined : tab,
  };
}

function sameSearch(a: ViewerSearch, b: ViewerSearch): boolean {
  return a.project === b.project && a.runs === b.runs && a.tab === b.tab;
}

/**
 * Keeps project, selected runs and active tab in the URL. The URL hydrates the store
 * on load and on history navigation; store changes are written back with `replace`.
 */
export function useUrlSync(search: ViewerSearch) {
  const navigate = useNavigate();

  useEffect(() => {
    const state = useViewerStore.getState();
    const current = toSearch(state.projectId, state.selectedRunIds, state.tab);
    if (sameSearch(current, search)) return;
    state.hydrate({
      projectId: search.project ?? null,
      selectedRunIds: search.runs === undefined ? [] : search.runs.split(","),
      tab: search.tab ?? "charts",
    });
  }, [search]);

  useEffect(
    () =>
      useViewerStore.subscribe((state, previous) => {
        if (
          state.projectId === previous.projectId &&
          state.selectedRunIds === previous.selectedRunIds &&
          state.tab === previous.tab
        )
          return;
        void navigate({
          to: "/",
          search: toSearch(state.projectId, state.selectedRunIds, state.tab),
          replace: true,
        });
      }),
    [navigate],
  );
}
