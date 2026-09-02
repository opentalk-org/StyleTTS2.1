import { ArrowUpRight, Clock3, FolderKanban } from "lucide-react";
import { useMemo, useState } from "react";

import type { Project } from "@/shared/types";
import { Badge, Card, GroupLabel, SearchInput } from "@/shared/ui";

interface ProjectsProps {
  projects: Project[];
  loading?: boolean;
  onOpen: (id: string) => void;
}


const GRID =
  "grid min-w-[900px] grid-cols-[minmax(240px,2fr)_80px_120px_140px_110px_32px] items-center gap-3 px-4";

export function Projects({ projects, loading = false, onOpen }: ProjectsProps) {
  const [query, setQuery] = useState("");
  const filteredProjects = useMemo(() => {
    const normalizedQuery = query.toLowerCase();
    return projects.filter((project) =>
      `${project.name} ${project.description}`.toLowerCase().includes(normalizedQuery),
    );
  }, [projects, query]);

  return (
    <main className="min-h-dvh">
      <header className="flex h-14 items-center border-b border-line bg-elevated px-5">
        <h1 className="m-0 text-base font-semibold tracking-tight text-fg">Projects</h1>
      </header>
      <div className="mx-auto max-w-[1240px] p-6">
        <Card>
          <div className="flex h-14 items-center border-b border-line px-3">
            <SearchInput
              label="Search projects"
              value={query}
              onValue={setQuery}
              placeholder="Search projects"
              className="w-full max-w-80"
            />
          </div>
          <div role="table" aria-label="Projects" className="overflow-x-auto">
            <div role="row" className={`${GRID} h-9 border-b border-line bg-inset`}>
              {["Project", "Runs", "Active", "Last activity", "Created"].map((label) => (
                <GroupLabel key={label} role="columnheader">
                  {label}
                </GroupLabel>
              ))}
              <span />
            </div>
            {filteredProjects.map((project) => (
              <ProjectRow key={project.id} project={project} onOpen={onOpen} />
            ))}
            {loading ? (
              <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">Loading projects…</p>
            ) : null}
            {!loading && projects.length === 0 ? (
              <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">
                No projects yet. A project appears here once it has its first run.
              </p>
            ) : null}
            {!loading && projects.length > 0 && filteredProjects.length === 0 ? (
              <p className="m-0 px-4 py-10 text-center text-xs text-fg-muted">
                No project matches “{query}”.
              </p>
            ) : null}
          </div>
          <footer className="flex h-10 items-center border-t border-line px-4 font-mono text-xs tabular-nums text-fg-muted">
            {loading ? "…" : filteredProjects.length} projects
          </footer>
        </Card>
      </div>
    </main>
  );
}

function ProjectRow({ project, onOpen }: { project: Project; onOpen: (id: string) => void }) {
  return (
    <button
      type="button"
      role="row"
      onClick={() => onOpen(project.id)}
      className={`${GRID} group h-16 w-full border-b border-line text-left text-sm text-fg-secondary transition-colors duration-150 ease-out last:border-b-0 hover:bg-surface`}
    >
      <span role="cell" className="flex min-w-0 items-center gap-3">
        <span className="grid size-8 shrink-0 place-items-center rounded-lg border border-line bg-inset text-fg-secondary transition-colors group-hover:border-accent-border group-hover:text-accent-bright">
          <FolderKanban size={15} />
        </span>
        <span className="flex min-w-0 flex-col">
          <strong className="truncate text-sm font-medium text-fg">{project.name}</strong>
          <small className="truncate text-xs text-fg-muted">{project.description}</small>
        </span>
      </span>
      <span role="cell" className="font-mono text-sm tabular-nums text-fg">
        {project.runCount}
      </span>
      <span role="cell">
        {project.runningCount === 0 ? (
          <em className="text-fg-muted not-italic">—</em>
        ) : (
          <Badge tone="accent" icon={<span className="size-1.5 rounded-full bg-accent-bright" aria-hidden />}>
            {project.runningCount} running
          </Badge>
        )}
      </span>
      <span role="cell" className="flex items-center gap-1.5 text-xs text-fg-muted">
        <Clock3 size={12} className="shrink-0" />
        <span className="font-mono tabular-nums">{relativeTime(project.lastRunAt)}</span>
      </span>
      <span role="cell" className="font-mono text-xs tabular-nums text-fg-muted">
        {new Date(project.createdAt).toLocaleDateString()}
      </span>
      <span role="cell" className="text-fg-muted transition-colors group-hover:text-accent-bright">
        <ArrowUpRight size={16} />
      </span>
    </button>
  );
}

function relativeTime(value: number): string {
  if (value === 0) return "never";
  const hours = Math.max(1, Math.round((Date.now() - value) / 3.6e6));
  return hours < 24 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
}
