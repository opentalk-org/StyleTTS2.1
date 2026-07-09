import { SPEAKER_NAMES } from "@/features/voices/constants";
import { fmtClock } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import type { Segment } from "./api";
import { useEditor } from "./editorStore";

const ROW_COLS = "22px 30px 92px 126px 1fr auto";

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
  const { segSel, segChecked, playPos, select, toggleCheck, seek, playing, togglePlay, setSegText, setSegPhon, setSegVoice, deleteSeg, mergeNext } = useEditor();
  const sel = seg.id === segSel;
  const checked = segChecked.includes(seg.id);

  return (
    <div className="px-0.5 py-1.5">
      <div
        className={cn(
          "overflow-hidden rounded-lg border",
          sel ? "border-blue-200 bg-blue-50" : "border-line bg-panel",
        )}
      >
      <div
        onClick={() => { select(seg.id); seek(seg.start); }}
        className="grid cursor-pointer items-center gap-2.5 px-2.5 py-1.5"
        style={{ gridTemplateColumns: ROW_COLS }}
      >
        <div onClick={(e) => e.stopPropagation()} className="flex items-center justify-center">
          <input
            type="checkbox"
            title="Select for bulk actions"
            checked={checked}
            onChange={() => toggleCheck(seg.id)}
            className="h-3.5 w-3.5 cursor-pointer accent-blue-500"
          />
        </div>
        <div className={cn("text-[11px] font-bold tabular-nums", sel ? "text-blue-600" : "text-txt-mute")}>
          {String(index + 1).padStart(3, "0")}
        </div>
        <div className="font-mono text-[11px] leading-tight tabular-nums text-txt-dim">
          {fmtClock(seg.start)}
          <div className="text-txt-mute">{fmtClock(seg.end)}</div>
        </div>
        <div onClick={(e) => e.stopPropagation()} className="flex min-w-0 flex-col gap-1">
          <div className="relative">
            <select
              value={seg.speaker}
              onChange={(e) => setSegVoice(seg.id, e.target.value)}
              className="h-7 w-full appearance-none rounded-md bg-panel-2 pl-2 pr-5 text-[11.5px] font-semibold text-txt-dim outline-none"
            >
              <option value="">None</option>
              {SPEAKER_NAMES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <span className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-txt-mute">
              <Icon name="chevron-down" size={11} strokeWidth={2.4} />
            </span>
          </div>
          <span className="truncate px-1 text-[10.5px] font-semibold uppercase tracking-wide text-txt-mute">
            {segmentTypeLabel(seg.type_)}
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
      {seg.alignment?.length ? (
        <div onClick={(e) => e.stopPropagation()} className="flex flex-wrap items-center gap-1 border-t border-line bg-panel-2/40 px-2.5 py-1.5">
          <span className="mr-0.5 text-[9.5px] font-semibold uppercase tracking-wide text-txt-mute">Words</span>
          {seg.alignment.map((word, i) => {
            const current = playPos >= word.start && playPos <= word.end;
            return (
              <button
                key={`${word.start}-${i}`}
                onClick={() => seek(word.start)}
                title={`${fmtClock(word.start)} – ${fmtClock(word.end)}`}
                className={cn(
                  "rounded px-1.5 py-0.5 font-mono text-[10.5px]",
                  current ? "bg-blue-500 text-white" : "bg-panel-2 text-txt-dim hover:bg-blue-100 hover:text-blue-600",
                )}
              >
                {word.word}
              </button>
            );
          })}
        </div>
      ) : null}
      </div>
    </div>
  );
}

function segmentTypeLabel(type: string | undefined): string {
  if (!type || type === "manual") return "Manual";
  if (type === "whisper") return "Whisper";
  if (type === "parakeet") return "Parakeet";
  if (type === "canary") return "Canary";
  return type;
}
