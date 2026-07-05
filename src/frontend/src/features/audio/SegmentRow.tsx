import { SPEAKERS } from "@/mock/constants";
import { fmtClock } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import type { Segment } from "./api";
import { useEditor } from "./editorStore";

const ROW_COLS = "30px 92px 120px 1fr auto";

function IconBtn({ icon, title, danger, onClick }: { icon: "play" | "merge" | "trash"; title: string; danger?: boolean; onClick: () => void }) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        "flex h-7 w-7 items-center justify-center rounded-md text-txt-mute",
        danger ? "hover:bg-red-50 hover:text-red-500" : "hover:bg-panel-2",
      )}
    >
      <Icon name={icon} size={13} strokeWidth={2.2} />
    </button>
  );
}

export function SegmentRow({ seg, index, isLast }: { seg: Segment; index: number; isLast: boolean }) {
  const { segSel, select, seek, playing, togglePlay, setSegText, setSegPhon, setSegVoice, deleteSeg, mergeNext } = useEditor();
  const sel = seg.id === segSel;

  return (
    <div className="px-0.5 py-1.5">
      <div
        onClick={() => { select(seg.id); seek(seg.start); }}
        className={cn(
          "grid cursor-pointer items-center gap-2.5 rounded-lg border px-2.5 py-1.5",
          sel ? "border-blue-200 bg-blue-50" : "border-line bg-panel",
        )}
        style={{ gridTemplateColumns: ROW_COLS }}
      >
        <div className={cn("text-[11px] font-bold tabular-nums", sel ? "text-blue-600" : "text-txt-mute")}>
          {String(index + 1).padStart(3, "0")}
        </div>
        <div className="font-mono text-[11px] leading-tight tabular-nums text-txt-dim">
          {fmtClock(seg.start)}
          <div className="text-txt-mute">{fmtClock(seg.end)}</div>
        </div>
        <div onClick={(e) => e.stopPropagation()} className="relative">
          <select
            value={seg.speaker}
            onChange={(e) => setSegVoice(seg.id, e.target.value)}
            className="h-7 w-full appearance-none rounded-md bg-panel-2 pl-2 pr-5 text-[11.5px] font-semibold text-txt-dim outline-none"
          >
            {SPEAKERS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
          <span className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-txt-mute">
            <Icon name="chevron-down" size={11} strokeWidth={2.4} />
          </span>
        </div>
        <div onClick={(e) => e.stopPropagation()} className="flex min-w-0 flex-col gap-1">
          <input
            value={seg.text}
            onChange={(e) => setSegText(seg.id, e.target.value)}
            placeholder="Transcription…"
            className="h-[26px] w-full rounded-md border-[1.5px] border-transparent bg-panel-2 px-2 text-[12.5px] text-txt outline-none focus:border-blue-500"
          />
          <input
            value={seg.phon}
            onChange={(e) => setSegPhon(seg.id, e.target.value)}
            placeholder="phonemes…"
            className="h-6 w-full bg-transparent px-2 font-mono text-[11.5px] text-blue-600 outline-none"
          />
        </div>
        <div onClick={(e) => e.stopPropagation()} className="flex gap-px">
          <IconBtn icon="play" title="Play region" onClick={() => { seek(seg.start); if (!playing) togglePlay(); }} />
          {!isLast ? <IconBtn icon="merge" title="Merge with next" onClick={() => mergeNext(seg.id)} /> : null}
          <IconBtn icon="trash" title="Delete" danger onClick={() => deleteSeg(seg.id)} />
        </div>
      </div>
    </div>
  );
}
