import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Badge } from "@/shared/ui/Badge";
import { IconButton } from "@/shared/ui/IconButton";
import type { Checkpoint } from "@/mock/types";
import { TYPE_TONE } from "./logic";
import { useCheckpoints } from "./store";

/** Column template shared by the grouped checkpoint tables. */
export const CHECKPOINT_COLS = "minmax(180px,1.4fr) 116px 1fr 120px";

export function CheckpointRow({ checkpoint: c }: { checkpoint: Checkpoint }) {
  const { rename, remove } = useCheckpoints();

  const del = () =>
    askConfirm({
      title: "Delete checkpoint?",
      desc: `Delete "${c.name}". This permanently removes the artifact.`,
      danger: true,
      label: "Delete checkpoint",
      onConfirm: () => {
        remove(c.id);
        showToast("Checkpoint deleted", undefined, "error");
      },
    });
  const doRename = () => {
    const n = window.prompt("Rename checkpoint", c.name);
    if (n && n.trim()) rename(c.id, n.trim());
  };

  return (
    <div
      className="grid items-center gap-3.5 border-b border-line px-4 py-3 last:border-b-0"
      style={{ gridTemplateColumns: CHECKPOINT_COLS }}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="flex h-[30px] w-[30px] flex-none items-center justify-center rounded-[7px] bg-panel-2 text-txt-dim">
          <Icon name="box" size={15} strokeWidth={2} />
        </div>
        <span className="truncate font-mono text-[13px] font-semibold text-txt">{c.name}</span>
      </div>
      <div>
        <Badge tone={TYPE_TONE[c.type]}>{c.type}</Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2.5 text-[11.5px] text-txt-mute">
        {c.spkMode !== "—" ? <span>spk: {c.spkMode}</span> : null}
        {c.decoder !== "—" ? <span>dec: {c.decoder}</span> : null}
        {c.symbols ? (
          <button
            onClick={() => showToast(`${c.symbols} symbols copied`)}
            className="inline-flex items-center gap-1 rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-txt-dim cursor-pointer"
          >
            <Icon name="copy" size={11} strokeWidth={2} />
            {c.symbols} sym
          </button>
        ) : null}
      </div>
      <div className="flex justify-end gap-0.5">
        <IconButton icon="edit" title="Rename" onClick={doRename} />
        <IconButton icon="download" title="Download" onClick={() => showToast(`Downloading ${c.name}.pth`)} />
        <IconButton icon="trash" danger title="Delete" onClick={del} />
      </div>
    </div>
  );
}
