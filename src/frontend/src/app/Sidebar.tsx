import { runningCount, useJobs } from "../features/jobs/store";
import { Icon } from "../shared/icons";
import { cn } from "../shared/ui/cn";
import { NAV_ITEMS } from "./nav";
import { useNav } from "./navStore";

export function Sidebar() {
  const { screen, navCollapsed, go, toggleNav } = useNav();
  const running = useJobs((s) => runningCount(s.jobs));
  const expanded = !navCollapsed;

  return (
    <aside
      className="flex flex-none flex-col overflow-hidden border-r border-line bg-panel transition-[width] duration-200"
      style={{ width: expanded ? 232 : 64 }}
    >
      <div className="flex h-14 flex-none items-center gap-2.5 border-b border-line px-4">
        <div className="flex h-7 w-7 flex-none items-center justify-center rounded-md bg-blue-500 text-white">
          <Icon name="audio-lines" size={18} strokeWidth={2.4} />
        </div>
        {expanded ? (
          <div className="whitespace-nowrap text-[15px] font-extrabold tracking-tight text-txt">
            StyleTTS <span className="text-blue-500">Studio</span>
          </div>
        ) : null}
      </div>

      <nav className="flex flex-1 flex-col gap-0.5 overflow-y-auto overflow-x-hidden p-2.5">
        {NAV_ITEMS.map((item) => {
          const active = screen === item.id || (item.id === "audio" && screen === "editor");
          return (
            <button
              key={item.id}
              onClick={() => go(item.id)}
              title={item.label}
              className={cn(
                "flex h-10 w-full items-center gap-3 rounded-md border-0 px-3 text-[13.5px] font-semibold cursor-pointer transition-colors",
                active ? "bg-blue-500 text-white" : "bg-transparent text-txt-dim hover:bg-panel-2 hover:text-txt",
              )}
            >
              <span className="flex w-5 flex-none items-center justify-center">
                <Icon name={item.icon} size={19} strokeWidth={active ? 2.3 : 2} />
              </span>
              {expanded ? <span className="flex-1 whitespace-nowrap text-left">{item.label}</span> : null}
              {item.id === "jobs" && running > 0 ? (
                <span
                  className={cn(
                    "h-2 w-2 flex-none rounded-full",
                    active ? "bg-white" : "bg-amber-500",
                  )}
                  style={{ animation: "pulse-dot 1.4s infinite" }}
                />
              ) : null}
            </button>
          );
        })}
      </nav>

      <div className="flex-none border-t border-line p-2.5">
        <button
          onClick={toggleNav}
          title={expanded ? "Collapse" : "Expand"}
          className="flex h-[38px] w-full items-center gap-3 rounded-md border-0 bg-transparent px-3 text-[13px] font-semibold text-txt-mute cursor-pointer hover:bg-panel-2 hover:text-txt"
        >
          <span className="flex w-5 flex-none">
            <Icon name={expanded ? "chevrons-left" : "chevrons-right"} size={18} strokeWidth={2.2} />
          </span>
          {expanded ? <span className="whitespace-nowrap">Collapse</span> : null}
        </button>
      </div>
    </aside>
  );
}
