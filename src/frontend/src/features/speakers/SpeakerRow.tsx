import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { Icon } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import { IconButton } from "@/shared/ui/IconButton";
import type { Speaker } from "./api";
import { useSpeakerActions } from "./query";
import { useSpeakerFilters } from "./store";

function Checkbox({ on }: { on: boolean }) {
  return (
    <span
      className={cn(
        "flex h-[18px] w-[18px] flex-none items-center justify-center rounded",
        on ? "bg-blue-500" : "border-2 border-line-2 bg-panel",
      )}
    >
      {on ? <Icon name="check" size={12} strokeWidth={3} className="text-white" /> : null}
    </span>
  );
}

export function SpeakerRow({ speaker }: { speaker: Speaker }) {
  const { editId, set, selection, selectAllMatching, toggleSelect } = useSpeakerFilters();
  const { rename, remove } = useSpeakerActions();
  const editing = editId === speaker.id;
  const selected = selectAllMatching || !!selection[speaker.id];

  const del = () =>
    askConfirm({
      title: "Delete speaker?",
      desc: `Clear "${speaker.id}" from its audio and segments.`,
      danger: true,
      label: "Delete speaker",
      onConfirm: () => remove(speaker.id),
    });

  return (
    <div className="py-1">
      <div
        className={cn(
          "flex h-[58px] items-center gap-3 rounded-[9px] border px-3.5",
          selected ? "border-blue-300 bg-blue-50" : "border-line bg-panel",
        )}
      >
        <button onClick={() => toggleSelect(speaker.id)} className="flex" title="Select speaker">
          <Checkbox on={selected} />
        </button>
        <div className="flex h-[34px] w-[34px] flex-none items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
          <Icon name="mic" size={16} strokeWidth={2.2} />
        </div>
        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              defaultValue={speaker.id}
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  rename(speaker.id, (e.target as HTMLInputElement).value.trim() || speaker.id);
                  set({ editId: null });
                }
                if (e.key === "Escape") set({ editId: null });
              }}
              onBlur={(e) => {
                rename(speaker.id, e.target.value.trim() || speaker.id);
                set({ editId: null });
              }}
              className="h-[30px] w-full max-w-[320px] rounded-md border-2 border-blue-500 bg-panel-2 px-2.5 text-[13.5px] font-semibold text-txt outline-none"
            />
          ) : (
            <>
              <div className="truncate text-[13.5px] font-semibold text-txt">{speaker.id}</div>
              <div className="mt-0.5 flex items-center gap-1.5">
                <span className="font-mono text-[11px] text-txt-mute">{speaker.audio_files.toLocaleString()} audio files</span>
              </div>
            </>
          )}
        </div>
        <span className="flex-none text-[12.5px] tabular-nums text-txt-dim">
          {speaker.segments.toLocaleString()} seg
        </span>
        <div className="flex flex-none gap-0.5">
          <IconButton icon="edit" title="Rename" onClick={() => set({ editId: editing ? null : speaker.id })} />
          <IconButton icon="trash" danger title="Delete" onClick={del} />
        </div>
      </div>
    </div>
  );
}
