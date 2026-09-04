import { getRouteApi } from "@tanstack/react-router";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { Viewer } from "@/features/viewer/Viewer";
import { useUrlSync } from "@/features/viewer/url";

const route = getRouteApi("/");

export function App() {
  const search = route.useSearch();
  useUrlSync(search);
  return (
    <ErrorBoundary>
      <Viewer />
    </ErrorBoundary>
  );
}
