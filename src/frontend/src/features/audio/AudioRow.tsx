import { useNav } from "@/app/navStore";
import { fmtAgo, fmtDur } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { WaveformBars } from "@/shared/media/WaveformBars";
import { cn } from "@/shared/ui/cn";
import { IconButton } from "@/shared/ui/IconButton";
import { InlineSegments } from "./InlineSegments";
import type { AudioFile } from "./api";
import { useAudio } from "./store";

export const AUDIO_COLS = "30px 22px 66px minmax(130px,1.2fr) 118px 54px 46px 74px 30px";

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

export function AudioRow({ file, index }: { file: AudioFile; index: number }) {
  const { selection, selectAllMatching, expanded, toggleSelect, toggleExpanded } = useAudio();
  const on = selectAllMatching || !!selection[file.id];
  const ex = !!expanded[file.id];
  const noSeg = file.segments === 0;
  const updatedAt = Date.parse(file.updated_at);
  const openEditor = useNav((s) => s.openEditor);

  return (
    <div>
      <div
        onClick={() => toggleExpanded(file.id)}
        className={cn(
          "grid h-[52px] cursor-pointer items-center gap-3 px-3.5 transition-colors",
          ex ? "" : "border-b border-line",
          on ? "bg-blue-50" : "hover:bg-panel-2",
        )}
        style={{ gridTemplateColumns: AUDIO_COLS }}
      >
        <button onClick={(e) => { e.stopPropagation(); toggleSelect(file.id); }} className="flex">
          <Checkbox on={on} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); toggleExpanded(file.id); }}
          title={ex ? "Collapse" : "Show segments"}
          className={cn("flex h-[22px] w-[22px] items-center justify-center text-txt-mute transition-transform", ex && "rotate-90")}
        >
          <Icon name="chevron-down" size={15} strokeWidth={2.4} className="-rotate-90" />
        </button>
        <WaveformBars seed={index + 1} bars={20} height={22} />
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-mono text-[13px] font-semibold text-txt">{file.name}</span>
          {noSeg ? (
            <span className="flex flex-none text-amber-500" title="No segments — Split or Transcribe to create them">
              <Icon name="alert" size={15} strokeWidth={2.2} />
            </span>
          ) : null}
        </div>
        <span className="truncate text-[12.5px] text-txt-dim">{file.speaker}</span>
        <span className="font-mono text-[12.5px] tabular-nums text-txt-dim">{fmtDur(file.duration)}</span>
        <span className={cn("justify-self-end text-[12.5px] tabular-nums", file.segments ? "font-semibold text-txt" : "text-txt-mute")}>
          {file.segments || "—"}
        </span>
        <span className="text-xs text-txt-mute">{Number.isNaN(updatedAt) ? "-" : fmtAgo(updatedAt)}</span>
        <IconButton
          icon="edit"
          title="Open segment editor"
          size={26}
          onClick={(event) => {
            event.stopPropagation();
            openEditor(file.id);
          }}
        />
      </div>
      {ex ? <InlineSegments file={file} /> : null}
    </div>
  );
}
