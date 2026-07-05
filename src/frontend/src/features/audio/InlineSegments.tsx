import { useDatasetsQuery } from "@/features/datasets/query";
import { fmtClock } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { splitAction, transcribeAction } from "./actions";
import type { AudioFile } from "./api";

const CAP = 8;

export function InlineSegments({ file }: { file: AudioFile }) {
  const segs = file.segment_preview;
  const { data: datasets = [] } = useDatasetsQuery();

  if (segs.length === 0)
    return (
      <div className="flex items-center gap-3 border-b border-line bg-gray-50 py-3 pl-[66px] pr-3.5">
        <span className="flex-none text-amber-500">
          <Icon name="alert" size={16} strokeWidth={2} />
        </span>
        <span className="flex-1 text-[12.5px] text-txt-dim">
          No segments yet. Split this file by silence or transcribe it to generate segments.
        </span>
        <Button variant="primary" size="sm" icon="scissors" onClick={() => splitAction(1, datasets)}>
          Split…
        </Button>
        <Button variant="ghost" size="sm" icon="file-audio" onClick={() => transcribeAction(1)}>
          Transcribe
        </Button>
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
            <div className="truncate text-[12.5px] text-txt">{g.text}</div>
            <div className="mt-px truncate font-mono text-[11.5px] text-blue-600">{g.phon}</div>
          </div>
        </div>
      ))}
      {file.segments > CAP ? <span className="ml-2 mt-1 text-xs font-semibold text-txt-mute">+ {file.segments - CAP} more segments</span> : null}
    </div>
  );
}
