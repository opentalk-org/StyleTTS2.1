import { useEffect, useRef } from "react";

import { useQueryClient } from "@tanstack/react-query";
import { Outlet } from "react-router-dom";

import { ConfirmHost } from "@/shared/feedback/ConfirmDialog";
import { ParamModalHost } from "@/shared/feedback/ParamModal";
import { ToastHost } from "@/shared/feedback/Toast";
import { ConnectScreen } from "./ConnectScreen";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { useAppStore } from "./store";

export function AppShell() {
  const connected = useAppStore((state) => state.connected);
  const backendUrl = useAppStore((state) => state.backendUrl);
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
          <Outlet />
        </main>
      </div>
      <ToastHost />
      <ConfirmHost />
      <ParamModalHost />
    </div>
  );
}
