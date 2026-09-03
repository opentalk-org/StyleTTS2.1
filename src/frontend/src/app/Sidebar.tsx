import { NavLink } from "react-router-dom";

import { Icon, type IconName } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import { useAppStore } from "./store";

function SidebarLink({ to, label, icon, expanded }: { to: string; label: string; icon: IconName; expanded: boolean }) {
  return (
    <NavLink to={to} title={label} className={({ isActive }) => cn(
      "flex h-10 w-full items-center gap-3 rounded-md px-3 text-[13.5px] font-semibold transition-colors",
      isActive ? "bg-blue-500 text-white" : "bg-transparent text-txt-dim hover:bg-panel-2 hover:text-txt",
    )}>
      {({ isActive }) => <>
        <span className="flex w-5 flex-none items-center justify-center"><Icon name={icon} size={19} strokeWidth={isActive ? 2.3 : 2} /></span>
        {expanded ? <span className="flex-1 whitespace-nowrap text-left">{label}</span> : null}
      </>}
    </NavLink>
  );
}

export function Sidebar() {
  const { navCollapsed, toggleNav } = useAppStore();
  const expanded = !navCollapsed;

  return (
    <aside className="flex flex-none flex-col overflow-hidden border-r border-line bg-panel transition-[width] duration-200" style={{ width: expanded ? 232 : 64 }}>
      <div className="flex h-14 flex-none items-center gap-2.5 border-b border-line px-4">
        <div className="flex h-7 w-7 flex-none items-center justify-center rounded-md bg-blue-500 text-white"><Icon name="audio-lines" size={18} strokeWidth={2.4} /></div>
        {expanded ? <div className="whitespace-nowrap text-[15px] font-extrabold tracking-tight text-txt">StyleTTS <span className="text-blue-500">Studio</span></div> : null}
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto overflow-x-hidden p-2.5">
        <SidebarLink to="/datasets" label="Datasets" icon="database" expanded={expanded} />
        <SidebarLink to="/speakers" label="Speakers" icon="mic" expanded={expanded} />
        <SidebarLink to="/audio" label="Audio Files" icon="audio-lines" expanded={expanded} />
        <SidebarLink to="/mos" label="MOS" icon="gauge" expanded={expanded} />
        <SidebarLink to="/statistics" label="Statistics" icon="bar-chart" expanded={expanded} />
        <SidebarLink to="/artifacts" label="Artifacts" icon="sparkles" expanded={expanded} />
        <SidebarLink to="/workflows" label="Workflows" icon="workflow" expanded={expanded} />
        <SidebarLink to="/checkpoints" label="Checkpoints" icon="box" expanded={expanded} />
        <SidebarLink to="/training" label="Training" icon="sliders" expanded={expanded} />
        <SidebarLink to="/runs" label="Runs" icon="activity" expanded={expanded} />
        <SidebarLink to="/testing" label="Testing" icon="flask" expanded={expanded} />
        <SidebarLink to="/cluster" label="Cluster" icon="server" expanded={expanded} />
        <SidebarLink to="/jobs" label="Jobs" icon="list-checks" expanded={expanded} />
        <SidebarLink to="/settings" label="Settings" icon="settings" expanded={expanded} />
      </nav>
      <div className="flex-none border-t border-line p-2.5">
        <button onClick={toggleNav} title={expanded ? "Collapse" : "Expand"} className="flex h-[38px] w-full items-center gap-3 rounded-md border-0 bg-transparent px-3 text-[13px] font-semibold text-txt-mute cursor-pointer hover:bg-panel-2 hover:text-txt">
          <span className="flex w-5 flex-none"><Icon name={expanded ? "chevrons-left" : "chevrons-right"} size={18} strokeWidth={2.2} /></span>
          {expanded ? <span className="whitespace-nowrap">Collapse</span> : null}
        </button>
      </div>
    </aside>
  );
}
