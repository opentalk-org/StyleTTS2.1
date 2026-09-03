import { BookmarkPlus, Trash2, X } from "lucide-react";

import type { Workspace } from "@/shared/types";
import { Button, GroupLabel, IconButton, Modal } from "@/shared/ui";

import { useViewerStore } from "./store";

interface ViewsDialogProps {
  open: boolean;
  projectName: string;
  onClose: () => void;
}

export function ViewsDialog({ open, projectName, onClose }: ViewsDialogProps) {
  const { workspaces, loadWorkspace, saveWorkspace, deleteWorkspace } = useViewerStore();

  function saveCurrentView() {
    const name = window.prompt("Name this view", `${projectName} comparison`);
    if (name !== null && name.length > 0) saveWorkspace(name);
  }

  function removeView(workspace: Workspace) {
    if (!window.confirm(`Delete the view “${workspace.name}”?`)) return;
    deleteWorkspace(workspace.id);
  }

  return (
    <Modal open={open} onClose={onClose} label="Views" centered className="max-w-2xl">
      <header className="flex h-14 flex-none items-center justify-between gap-3 border-b border-line px-4">
        <div className="flex min-w-0 flex-col gap-1">
          <GroupLabel>Saved configurations</GroupLabel>
          <h2 className="m-0 text-base leading-tight font-semibold tracking-tight text-fg">Views</h2>
        </div>
        <IconButton label="Close views" onClick={onClose}><X size={15} /></IconButton>
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        {workspaces.length === 0 ? (
          <p className="m-0 px-4 py-10 text-center text-xs leading-relaxed text-fg-muted">
            A view stores the plots, table columns, run colors and query you have set up.
            <br />
            Save one to come back to this exact setup later.
          </p>
        ) : workspaces.map((workspace) => (
          <div
            key={workspace.id}
            className="flex items-center gap-3 border-b border-line px-4 py-3 transition-colors duration-150 last:border-b-0 hover:bg-surface"
          >
            <div className="flex min-w-0 flex-1 flex-col gap-1">
              <strong className="truncate text-sm font-medium text-fg">{workspace.name}</strong>
              <span className="font-mono text-[11px] tabular-nums text-fg-muted">
                {workspace.columns.length} columns · {workspace.selectedRunIds.length} runs ·{" "}
                {new Date(workspace.updatedAt).toLocaleDateString()}
              </span>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                loadWorkspace(workspace.id);
                onClose();
              }}
            >
              Load
            </Button>
            <IconButton
              label={`Delete view ${workspace.name}`}
              size="sm"
              variant="secondary"
              className="hover:bg-negative-surface hover:text-negative"
              onClick={() => removeView(workspace)}
            >
              <Trash2 size={13} />
            </IconButton>
          </div>
        ))}
      </div>

      <footer className="flex flex-none items-center justify-between gap-3 border-t border-line p-3">
        <span className="font-mono text-xs tabular-nums text-fg-muted">{workspaces.length} saved</span>
        <Button variant="primary" icon={<BookmarkPlus size={14} />} onClick={saveCurrentView}>
          Save current view
        </Button>
      </footer>
    </Modal>
  );
}
