import { fmtClock } from "@/shared/format";
import { Icon } from "@/shared/icons";
import type { AudioFile } from "./api";

const CAP = 8;

export function InlineSegments({ file }: { file: AudioFile }) {
  const segs = file.segment_preview;

  if (segs.length === 0)
    return (
      <div className="flex items-center gap-3 border-b border-line bg-gray-50 py-3 pl-[66px] pr-3.5">
        <span className="flex-none text-amber-500">
          <Icon name="alert" size={16} strokeWidth={2} />
        </span>
        <span className="flex-1 text-[12.5px] text-txt-dim">
          No segments yet. Split this file by silence or transcribe it to generate segments.
        </span>
      </div>
    );

  const shown = segs.slice(0, CAP);
  return (
    <div className="flex flex-col gap-1 border-b border-line bg-gray-50 py-2.5 pb-3 pl-[66px] pr-3.5">
      {shown.map((g, i) => (
        <div
          key={g.id}
          className="grid items-center gap-3 rounded-md px-2 py-1.5 hover:bg-panel"
          style={{ gridTemplateColumns: "26px 96px 1fr" }}
        >
          <span className="text-[11px] font-bold tabular-nums text-txt-mute">
            {String(i + 1).padStart(2, "0")}
          </span>
          <span className="font-mono text-[11px] tabular-nums text-txt-mute">
            {fmtClock(g.start)}-{fmtClock(g.end)}
          </span>
          <div className="min-w-0">
            <div className="flex min-w-0 items-center gap-2">
              <span className="truncate text-[12.5px] text-txt">{g.text}</span>
              <span className="flex-none rounded bg-panel-2 px-1.5 py-0.5 text-[10px] font-semibold text-txt-mute">{segmentTypeLabel(g.type_)}</span>
            </div>
            <div className="mt-px truncate font-mono text-[11.5px] text-blue-600">{g.phon}</div>
          </div>
        </div>
      ))}
      {file.segments > CAP ? <span className="ml-2 mt-1 text-xs font-semibold text-txt-mute">+ {file.segments - CAP} more segments</span> : null}
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
