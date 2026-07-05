import { VirtualTable } from "@/shared/data/VirtualTable";
import { Icon } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import { AUDIO_COLS, AudioRow } from "./AudioRow";
import { AudioToolbar } from "./AudioToolbar";
import { SelectionBar } from "./SelectionBar";
import { filteredAudioCount } from "./logic";
import { useAudio } from "./store";

const HEAD = ["Wave", "File", "Voice", "Dur"];

function Header({ allSel, onToggleAll }: { allSel: boolean; onToggleAll: () => void }) {
  return (
    <div
      className="sticky top-0 z-[2] grid h-10 items-center gap-3 border-b border-line bg-panel px-3.5"
      style={{ gridTemplateColumns: AUDIO_COLS }}
    >
      <button onClick={onToggleAll} className="flex">
        <span className={cn("flex h-[18px] w-[18px] items-center justify-center rounded", allSel ? "bg-blue-500" : "border-2 border-line-2 bg-panel")}>
          {allSel ? <Icon name="check" size={12} strokeWidth={3} className="text-white" /> : null}
        </span>
      </button>
      <span />
      {HEAD.map((h) => (
        <span key={h} className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">
          {h}
        </span>
      ))}
      <span className="justify-self-end text-[11px] font-bold uppercase tracking-wider text-txt-mute">Segs</span>
      <span className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">Updated</span>
    </div>
  );
}

export function AudioScreen() {
  const { selection, selectAllMatching, selectAllFiltered, clearSelection } = useAudio();
  const count = filteredAudioCount();
  const hasSelection = selectAllMatching || Object.keys(selection).length > 0;

  return (
    <div className="mx-auto flex h-full max-w-[1240px] flex-col px-7 pb-6 pt-5">
      <AudioToolbar />
      {hasSelection ? <SelectionBar /> : null}
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[10px] border border-line bg-panel">
        <VirtualTable
          count={count}
          estimateRowHeight={52}
          className="flex-1"
          header={<Header allSel={selectAllMatching} onToggleAll={selectAllMatching ? clearSelection : selectAllFiltered} />}
          renderRow={(i) => <AudioRow index={i} />}
        />
      </div>
      <div className="mt-3 text-[12.5px] tabular-nums text-txt-mute">{count.toLocaleString()} files</div>
    </div>
  );
}
