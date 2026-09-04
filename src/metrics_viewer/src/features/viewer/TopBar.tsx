import { ChevronDown, ChevronRight, CircleHelp, Moon, RefreshCw, Sun } from "lucide-react";
import { useState } from "react";

import type { Project } from "@/shared/types";
import { IconButton, MenuItem, Popover } from "@/shared/ui";

import { useViewerStore } from "./store";
import { ViewsMenu } from "./ViewsMenu";

interface TopBarProps {
  projects: Project[];
  project: Project | undefined;
  theme: "dark" | "light";
  onTheme: () => void;
  onHelp: () => void;
  anyRunning: boolean;
  onRefresh: () => void;
}

export function TopBar({ projects, project, theme, onTheme, onHelp, anyRunning, onRefresh }: TopBarProps) {
  const selectProject = useViewerStore((state) => state.selectProject);
  const [projectMenu, setProjectMenu] = useState(false);

  return (
    <header className="flex h-topbar flex-none items-center justify-between gap-3 border-b border-line bg-surface px-3">
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1 text-[13px]">
        <button
          type="button"
          onClick={() => selectProject(null)}
          className={
            project === undefined
              ? "rounded-sm px-1.5 py-0.5 font-semibold text-fg"
              : "rounded-sm px-1.5 py-0.5 text-fg-muted hover:bg-hover hover:text-fg"
          }
        >
          Projects
        </button>
        {project === undefined ? null : (
          <>
            <ChevronRight size={13} className="shrink-0 text-fg-muted" />
            <Popover
              open={projectMenu}
              onClose={() => setProjectMenu(false)}
              align="start"
              title="Switch project"
              width={280}
              trigger={
                <button
                  type="button"
                  aria-haspopup="menu"
                  aria-expanded={projectMenu}
                  onClick={() => setProjectMenu(!projectMenu)}
                  className="flex min-w-0 items-center gap-1 rounded-sm px-1.5 py-0.5 font-semibold text-fg hover:bg-hover"
                >
                  <span className="truncate">{project.name}</span>
                  <ChevronDown size={13} className="shrink-0 text-fg-muted" />
                </button>
              }
            >
              {projects.map((candidate) => (
                <MenuItem
                  key={candidate.id}
                  label={candidate.name}
                  hint={`${candidate.runCount} runs`}
                  onSelect={() => {
                    setProjectMenu(false);
                    if (candidate.id !== project.id) selectProject(candidate.id);
                  }}
                />
              ))}
            </Popover>
          </>
        )}
      </nav>

      <div className="flex shrink-0 items-center gap-1">
        {project === undefined ? null : <ViewsMenu projectName={project.name} />}
        {project === undefined ? null : (
          <IconButton label={anyRunning ? "Refresh runs (some are running)" : "Refresh runs"} onClick={onRefresh}>
            <RefreshCw size={14} className={anyRunning ? "text-accent" : undefined} />
          </IconButton>
        )}
        <IconButton label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} onClick={onTheme}>
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </IconButton>
        <IconButton label="Keyboard shortcuts" shortcut="?" onClick={onHelp}>
          <CircleHelp size={14} />
        </IconButton>
      </div>
    </header>
  );
}
