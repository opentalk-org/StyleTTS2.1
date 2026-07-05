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
