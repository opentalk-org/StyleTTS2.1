import { useEffect, useRef } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { ConfirmHost } from "@/shared/feedback/ConfirmDialog";
import { ParamModalHost } from "@/shared/feedback/ParamModal";
import { ToastHost } from "@/shared/feedback/Toast";
import { ConnectScreen } from "./ConnectScreen";
import { Header } from "./Header";
import { ScreenRouter } from "./ScreenRouter";
import { Sidebar } from "./Sidebar";
import { useNav } from "./navStore";

export function AppShell() {
  const connected = useNav((s) => s.connected);
  const backendUrl = useNav((s) => s.backendUrl);
  const queryClient = useQueryClient();
  const previousBackendUrl = useRef(backendUrl);

  useEffect(() => {
    if (!connected) {
      previousBackendUrl.current = backendUrl;
      return;
    }
    if (previousBackendUrl.current === backendUrl) return;
    previousBackendUrl.current = backendUrl;
    queryClient.invalidateQueries();
  }, [backendUrl, connected, queryClient]);

  if (!connected) return <ConnectScreen />;

  return (
    <div className="flex h-screen w-full overflow-hidden bg-app">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Header />
        <main className="relative flex-1 overflow-y-auto overflow-x-hidden">
          <ScreenRouter />
        </main>
      </div>
      <ToastHost />
      <ConfirmHost />
      <ParamModalHost />
    </div>
  );
}
