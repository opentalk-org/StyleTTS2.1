import { useLocation } from "react-router-dom";

import { Icon, type IconName } from "@/shared/icons";
import { useAppStore } from "./store";

function screenMetadata(pathname: string): { title: string; icon: IconName } {
  if (pathname.startsWith("/audio/")) return { title: "Segment Editor", icon: "audio-lines" };
  switch (pathname.split("/")[1]) {
    case "datasets": return { title: "Datasets", icon: "database" };
    case "speakers": return { title: "Speakers", icon: "mic" };
    case "audio": return { title: "Audio Files", icon: "audio-lines" };
    case "mos": return { title: "MOS", icon: "gauge" };
    case "statistics": return { title: "Statistics", icon: "bar-chart" };
    case "artifacts": return { title: "Artifacts", icon: "sparkles" };
    case "workflows": return { title: "Workflows", icon: "workflow" };
    case "checkpoints": return { title: "Checkpoints", icon: "box" };
    case "training": return { title: "Training", icon: "sliders" };
    case "runs": return { title: "Runs", icon: "activity" };
    case "testing": return { title: "Testing", icon: "flask" };
    case "cluster": return { title: "Cluster", icon: "server" };
    case "jobs": return { title: "Jobs", icon: "list-checks" };
    case "settings": return { title: "Settings", icon: "settings" };
    default: return { title: "Training", icon: "sliders" };
  }
}

export function Header() {
  const backendUrl = useAppStore((state) => state.backendUrl);
  const meta = screenMetadata(useLocation().pathname);

  return (
    <header className="flex h-14 flex-none items-center gap-4 border-b border-line bg-panel px-5">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="flex text-txt-mute"><Icon name={meta.icon} size={18} /></span>
        <div className="whitespace-nowrap text-[15px] font-bold tracking-tight text-txt">{meta.title}</div>
      </div>
      <div className="flex-1" />
      <div className="flex h-8 items-center gap-2 rounded-full bg-panel-2 px-3 text-xs font-medium text-txt-dim">
        <span className="h-[7px] w-[7px] rounded-full bg-emerald-500" />
        <span className="tabular-nums">{backendUrl.replace(/^https?:\/\//, "")}</span>
      </div>
    </header>
  );
}
