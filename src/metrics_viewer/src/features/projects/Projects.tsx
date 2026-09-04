import { FolderKanban } from "lucide-react";
import { useMemo, useState } from "react";

import { formatRelative } from "@/features/runs/logic";
import type { Project } from "@/shared/types";
import { Badge, Caption, cn, EmptyState, SearchInput, Skeleton, StatusMark, Tooltip } from "@/shared/ui";

interface ProjectsProps {
  projects: Project[];
  loading?: boolean;
  onOpen: (id: string) => void;
}

const GRID = "grid grid-cols-[minmax(240px,1fr)_88px_120px_120px_110px] items-center gap-3 px-4";

export function Projects({ projects, loading = false, onOpen }: ProjectsProps) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const normalized = query.toLowerCase();
    return projects.filter((project) => `${project.name} ${project.description}`.toLowerCase().includes(normalized));
  }, [projects, query]);

  return (
    <div className="min-h-0 flex-1 overflow-auto">
      <div className="mx-auto flex max-w-[1200px] flex-col gap-4 px-6 py-6">
        <div className="flex items-center justify-between gap-4">
          <h1 className="text-lg font-semibold text-fg">Projects</h1>
          <SearchInput
            label="Search projects"
            value={query}
            onValue={setQuery}
            placeholder="Search projects"
            shortcut="/"
            data-shortcut="search"
            size="lg"
            className="w-72"
          />
        </div>

        <div role="table" aria-label="Projects" className="overflow-x-auto rounded-md border border-line bg-surface">
          <div role="row" className={cn(GRID, "h-thead border-b border-line")}>
            {["Name", "Runs", "Running", "Last activity", "Created"].map((label, index) => (
              <Caption key={label} role="columnheader" className={index === 1 ? "text-right" : ""}>
                {label}
              </Caption>
            ))}
          </div>
          {loading ? (
            <div className="flex flex-col">
              {Array.from({ length: 5 }, (_, index) => (
                <div key={index} className={cn(GRID, "h-10 border-b border-line last:border-b-0")}>
                  <Skeleton className="h-3 w-48" />
                  <Skeleton className="ml-auto h-3 w-8" />
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-3 w-12" />
                  <Skeleton className="h-3 w-16" />
                </div>
              ))}
            </div>
          ) : null}
          {!loading && projects.length === 0 ? (
            <EmptyState compact icon={<FolderKanban />} title="No projects yet" description="A project appears here once it has its first run." />
          ) : null}
          {!loading && projects.length > 0 && filtered.length === 0 ? (
            <EmptyState compact icon={<FolderKanban />} title="No project matches" description={`Nothing contains “${query}”.`} />
          ) : null}
          {filtered.map((project) => (
            <ProjectRow key={project.id} project={project} onOpen={onOpen} />
          ))}
        </div>
        <span className="font-mono text-xs tabular-nums text-fg-muted">
          {loading ? "…" : `${filtered.length} projects`}
        </span>
      </div>
    </div>
  );
}

function ProjectRow({ project, onOpen }: { project: Project; onOpen: (id: string) => void }) {
  const hasDescription = project.description.length > 0;
  return (
    <button
      type="button"
      role="row"
      onClick={() => onOpen(project.id)}
      className={cn(
        GRID,
        "w-full border-b border-line text-left text-[13px] last:border-b-0 hover:bg-hover",
        hasDescription ? "h-[52px]" : "h-10",
      )}
    >
      <span role="cell" className="flex min-w-0 flex-col">
        <span className="truncate font-medium text-fg">{project.name}</span>
        {hasDescription ? <span className="truncate text-xs text-fg-muted">{project.description}</span> : null}
      </span>
      <span role="cell" className="text-right font-mono text-xs tabular-nums text-fg-secondary">
        {project.runCount}
      </span>
      <span role="cell">
        {project.runningCount === 0 ? (
          <span className="text-fg-muted">—</span>
        ) : (
          <Badge tone="accent" icon={<StatusMark status="running" />}>
            {project.runningCount} running
          </Badge>
        )}
      </span>
      <span role="cell" className="text-xs text-fg-secondary">
        <Tooltip content={project.lastRunAt === 0 ? "No runs" : new Date(project.lastRunAt).toLocaleString()}>
          <span>{formatRelative(project.lastRunAt)}</span>
        </Tooltip>
      </span>
      <span role="cell" className="font-mono text-xs tabular-nums text-fg-muted">
        {new Date(project.createdAt).toLocaleDateString()}
      </span>
    </button>
  );
}
