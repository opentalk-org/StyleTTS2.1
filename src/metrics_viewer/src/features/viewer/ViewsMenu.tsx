import { Bookmark, BookmarkPlus, Check, Pencil, Trash2, X } from "lucide-react";
import { useState } from "react";

import type { Workspace } from "@/shared/types";
import { Button, cn, IconButton, Popover, TextInput } from "@/shared/ui";

import { useViewerStore } from "./store";

export function ViewsMenu({ projectName }: { projectName: string }) {
  const { workspaces, projectId, loadWorkspace, saveWorkspace, renameWorkspace, deleteWorkspace } =
    useViewerStore();
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const views = workspaces.filter((workspace) => workspace.projectId === projectId);

  function startSave() {
    setName(`${projectName} · ${new Date().toLocaleDateString()}`);
    setSaving(true);
  }

  function commitSave() {
    const trimmed = name.trim();
    if (trimmed.length === 0) return;
    saveWorkspace(trimmed);
    setSaving(false);
  }

  return (
    <Popover
      open={open}
      onClose={() => setOpen(false)}
      title="Saved views"
      width={340}
      trigger={
        <Button variant="ghost" icon={<Bookmark size={14} />} aria-expanded={open} onClick={() => setOpen(!open)}>
          Views
          {views.length > 0 ? <span className="font-mono text-[11px] text-fg-muted">{views.length}</span> : null}
        </Button>
      }
    >
      {views.length === 0 && !saving ? (
        <p className="px-2 py-4 text-center text-xs leading-relaxed text-fg-muted">
          A view stores the selected runs, columns, colours, chart settings and query.
        </p>
      ) : null}
      {views.map((workspace) => (
        <ViewRow
          key={workspace.id}
          workspace={workspace}
          editing={editingId === workspace.id}
          confirming={confirmId === workspace.id}
          onLoad={() => {
            loadWorkspace(workspace.id);
            setOpen(false);
          }}
          onEdit={() => setEditingId(workspace.id)}
          onRename={(value) => {
            renameWorkspace(workspace.id, value);
            setEditingId(null);
          }}
          onCancelEdit={() => setEditingId(null)}
          onAskDelete={() => setConfirmId(workspace.id)}
          onDelete={() => {
            deleteWorkspace(workspace.id);
            setConfirmId(null);
          }}
          onCancelDelete={() => setConfirmId(null)}
        />
      ))}
      <div className="mt-1 border-t border-line pt-1">
        {saving ? (
          <form
            className="flex items-center gap-1 p-1"
            onSubmit={(event) => {
              event.preventDefault();
              commitSave();
            }}
          >
            <TextInput autoFocus value={name} onValue={setName} aria-label="View name" className="flex-1" />
            <IconButton label="Save" type="submit" size="sm" variant="primary">
              <Check size={13} />
            </IconButton>
            <IconButton label="Cancel" size="sm" onClick={() => setSaving(false)}>
              <X size={13} />
            </IconButton>
          </form>
        ) : (
          <Button variant="ghost" icon={<BookmarkPlus size={14} />} className="w-full justify-start" onClick={startSave}>
            Save current view
          </Button>
        )}
      </div>
    </Popover>
  );
}

interface ViewRowProps {
  workspace: Workspace;
  editing: boolean;
  confirming: boolean;
  onLoad: () => void;
  onEdit: () => void;
  onRename: (name: string) => void;
  onCancelEdit: () => void;
  onAskDelete: () => void;
  onDelete: () => void;
  onCancelDelete: () => void;
}

function ViewRow({
  workspace,
  editing,
  confirming,
  onLoad,
  onEdit,
  onRename,
  onCancelEdit,
  onAskDelete,
  onDelete,
  onCancelDelete,
}: ViewRowProps) {
  const [draft, setDraft] = useState(workspace.name);

  if (editing) {
    return (
      <form
        className="flex items-center gap-1 p-1"
        onSubmit={(event) => {
          event.preventDefault();
          if (draft.trim().length > 0) onRename(draft.trim());
        }}
      >
        <TextInput autoFocus value={draft} onValue={setDraft} aria-label="Rename view" className="flex-1" />
        <IconButton label="Save name" type="submit" size="sm" variant="primary">
          <Check size={13} />
        </IconButton>
        <IconButton label="Cancel" size="sm" onClick={onCancelEdit}>
          <X size={13} />
        </IconButton>
      </form>
    );
  }

  return (
    <div className={cn("group flex items-center gap-1 rounded-sm px-1 py-1 hover:bg-hover", confirming ? "bg-hover" : "")}>
      <button type="button" onClick={onLoad} className="flex min-w-0 flex-1 flex-col items-start text-left">
        <span className="w-full truncate text-[13px] font-medium text-fg">{workspace.name}</span>
        <span className="font-mono text-[11px] text-fg-muted">
          {workspace.selectedRunIds.length} runs · {workspace.columns.length} columns ·{" "}
          {new Date(workspace.updatedAt).toLocaleDateString()}
        </span>
      </button>
      {confirming ? (
        <>
          <Button size="sm" variant="ghost" className="text-failed hover:bg-failed-bg" onClick={onDelete}>
            Delete
          </Button>
          <Button size="sm" variant="ghost" onClick={onCancelDelete}>
            Keep
          </Button>
        </>
      ) : (
        <span className="flex items-center opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
          <IconButton label="Rename" size="sm" onClick={onEdit}>
            <Pencil size={12} />
          </IconButton>
          <IconButton label="Delete" size="sm" onClick={onAskDelete}>
            <Trash2 size={12} />
          </IconButton>
        </span>
      )}
    </div>
  );
}
