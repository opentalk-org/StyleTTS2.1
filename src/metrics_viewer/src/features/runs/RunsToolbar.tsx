import { Columns3, Ellipsis, Filter, PanelLeftClose, Rows2, Columns2, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";

import type { ViewerLayout } from "@/features/viewer/layout";
import { projectColumnOptions } from "@/shared/metrics";
import type { ProjectColumns, Run, RunStatus } from "@/shared/types";
import {
  Checkbox,
  IconButton,
  MenuItem,
  MenuSeparator,
  Popover,
  SearchInput,
  SearchOptionList,
  statusLabel,
  StatusMark,
} from "@/shared/ui";

import { STATUS_ORDER } from "./logic";

interface RunsToolbarProps {
  runs: Run[];
  projectColumns: ProjectColumns;
  query: string;
  onQuery: (value: string) => void;
  statuses: RunStatus[];
  onStatuses: (value: RunStatus[]) => void;
  columns: string[];
  onColumns: (value: string[]) => void;
  layout: ViewerLayout;
  onCollapse: () => void;
  onStack: () => void;
  onResetLayout: () => void;
}

export function RunsToolbar({
  runs,
  projectColumns,
  query,
  onQuery,
  statuses,
  onStatuses,
  columns,
  onColumns,
  layout,
  onCollapse,
  onStack,
  onResetLayout,
}: RunsToolbarProps) {
  const [statusOpen, setStatusOpen] = useState(false);
  const [columnsOpen, setColumnsOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const columnOptions = useMemo(() => projectColumnOptions(projectColumns), [projectColumns]);
  const counts = useMemo(() => {
    const byStatus = new Map<RunStatus, number>();
    for (const run of runs) byStatus.set(run.status, (byStatus.get(run.status) ?? 0) + 1);
    return byStatus;
  }, [runs]);

  return (
    <div className="flex h-toolbar flex-none items-center gap-1.5 border-b border-line bg-surface px-3">
      <SearchInput
        label="Search runs"
        value={query}
        onValue={onQuery}
        placeholder="Search runs or params"
        shortcut="/"
        data-shortcut="search"
        className="min-w-0 flex-1"
      />
      <Popover
        open={statusOpen}
        onClose={() => setStatusOpen(false)}
        title="Filter by status"
        width={200}
        trigger={
          <IconButton
            label={statuses.length === 0 ? "Filter by status" : `Status: ${statuses.map(statusLabel).join(", ")}`}
            active={statuses.length > 0 || statusOpen}
            onClick={() => setStatusOpen(!statusOpen)}
          >
            <Filter size={14} />
          </IconButton>
        }
      >
        {STATUS_ORDER.map((status) => (
          <Checkbox
            key={status}
            checked={statuses.includes(status)}
            onChange={() =>
              onStatuses(
                statuses.includes(status) ? statuses.filter((item) => item !== status) : [...statuses, status],
              )
            }
          >
            <StatusMark status={status} className={`text-${status}`} />
            <span className="flex-1">{statusLabel(status)}</span>
            <span className="font-mono text-[11px] text-fg-muted">{counts.get(status) ?? 0}</span>
          </Checkbox>
        ))}
        {statuses.length === 0 ? null : (
          <>
            <MenuSeparator />
            <MenuItem label="Clear filter" onSelect={() => onStatuses([])} />
          </>
        )}
      </Popover>
      <Popover
        open={columnsOpen}
        onClose={() => setColumnsOpen(false)}
        width={300}
        panelClassName="[&>div]:p-0"
        trigger={
          <IconButton label="Choose columns" active={columnsOpen} onClick={() => setColumnsOpen(!columnsOpen)}>
            <Columns3 size={14} />
          </IconButton>
        }
      >
        <SearchOptionList
          multiple
          options={columnOptions}
          selected={columns}
          placeholder="Search columns"
          emptyMessage="No column matches"
          onSelect={(id) =>
            onColumns(columns.includes(id) ? columns.filter((item) => item !== id) : [...columns, id])
          }
        />
      </Popover>
      <Popover
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        width={220}
        trigger={
          <IconButton label="Layout" active={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>
            <Ellipsis size={14} />
          </IconButton>
        }
      >
        <MenuItem
          icon={<PanelLeftClose />}
          label="Collapse runs"
          onSelect={() => {
            setMenuOpen(false);
            onCollapse();
          }}
        />
        <MenuItem
          icon={layout.orientation === "columns" ? <Rows2 /> : <Columns2 />}
          label={layout.orientation === "columns" ? "Stack panes vertically" : "Place panes side by side"}
          onSelect={() => {
            setMenuOpen(false);
            onStack();
          }}
        />
        <MenuSeparator />
        <MenuItem
          icon={<RotateCcw />}
          label="Reset layout"
          onSelect={() => {
            setMenuOpen(false);
            onResetLayout();
          }}
        />
      </Popover>
    </div>
  );
}
