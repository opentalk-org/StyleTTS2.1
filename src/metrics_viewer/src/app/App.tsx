import { getRouteApi } from "@tanstack/react-router";
import { useEffect } from "react";

import { ErrorBoundary } from "@/app/ErrorBoundary";
import { Viewer } from "@/features/viewer/Viewer";
import { useUrlSync } from "@/features/viewer/url";

const route = getRouteApi("/");

export function App() {
  const search = route.useSearch();
  useUrlSync(search);
  useReloadAfterDevServerRestart();
  return (
    <ErrorBoundary>
      <Viewer />
    </ErrorBoundary>
  );
}

/** A restarted Start server cannot execute IDs held by a client bundle from its predecessor. */
function useReloadAfterDevServerRestart() {
  useEffect(() => {
    const hot = import.meta.hot;
    if (hot === undefined) return;
    let disconnected = false;
    const markDisconnected = () => {
      disconnected = true;
    };
    const reloadAfterReconnect = () => {
      if (disconnected) window.location.reload();
    };
    hot.on("vite:ws:disconnect", markDisconnected);
    hot.on("vite:ws:connect", reloadAfterReconnect);
    return () => {
      hot.off("vite:ws:disconnect", markDisconnected);
      hot.off("vite:ws:connect", reloadAfterReconnect);
    };
  }, []);
}
