import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { Icon } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import type { Voice } from "./api";
import { useVoiceActions } from "./query";
import { useVoiceFilters } from "./store";

export function VoiceRow({ voice }: { voice: Voice }) {
  const { editId, set } = useVoiceFilters();
  const { rename, remove } = useVoiceActions();
  const editing = editId === voice.id;

  const del = () =>
    askConfirm({
      title: "Delete voice?",
      desc: `Delete "${voice.name}". Segments keep their text but lose this voice label.`,
      danger: true,
      label: "Delete voice",
      onConfirm: () => remove(voice.id),
    });

  return (
    <div className="py-1">
      <div className="flex h-[58px] items-center gap-3 rounded-[9px] border border-line bg-panel px-3.5">
        <div className="flex h-[34px] w-[34px] flex-none items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
          <Icon name="mic" size={16} strokeWidth={2.2} />
        </div>
        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              defaultValue={voice.name}
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  rename(voice.id, (e.target as HTMLInputElement).value.trim() || voice.name);
                  set({ editId: null });
                }
                if (e.key === "Escape") set({ editId: null });
              }}
              onBlur={(e) => {
                rename(voice.id, e.target.value.trim() || voice.name);
                set({ editId: null });
              }}
              className="h-[30px] w-full max-w-[320px] rounded-md border-2 border-blue-500 bg-panel-2 px-2.5 text-[13.5px] font-semibold text-txt outline-none"
            />
          ) : (
            <>
              <div className="truncate text-[13.5px] font-semibold text-txt">{voice.name}</div>
              <div className="mt-0.5 flex items-center gap-1.5">
                <span className="font-mono text-[11px] text-txt-mute">{voice.id}</span>
              </div>
            </>
          )}
        </div>
        <span className="flex-none text-[12.5px] tabular-nums text-txt-dim">
          {voice.segments.toLocaleString()} seg
        </span>
        <div className="flex flex-none gap-0.5">
          <IconButton icon="edit" title="Rename" onClick={() => set({ editId: editing ? null : voice.id })} />
          <IconButton icon="trash" danger title="Delete" onClick={del} />
        </div>
      </div>
    </div>
  );
}
