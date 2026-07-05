import { useState } from "react";

import { useDatasetsQuery } from "@/features/datasets/query";
import { Icon, type IconName } from "@/shared/icons";
import { cn } from "@/shared/ui/cn";
import {
  addDatasetAction,
  assignVoiceAction,
  calculateStatisticsAction,
  deleteFilesAction,
  denoiseAction,
  normalizeAction,
  phonemizeAction,
  removeDatasetAction,
  removeSegmentsAction,
  splitAction,
  transcribeAction,
} from "./actions";
import { useAudio } from "./store";

type MenuName = "process" | "dataset";

type Item =
  | { header: string }
  | { divider: true }
  | { label: string; icon: IconName; onClick: () => void; danger?: boolean };

function DarkButton({ label, icon, open, onClick }: { label: string; icon: IconName; open: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex h-[34px] items-center gap-1.5 rounded-md px-3 text-[12.5px] font-semibold text-white",
        open ? "bg-white/20" : "bg-white/10 hover:bg-white/15",
      )}
    >
      <Icon name={icon} size={15} strokeWidth={2.2} />
      {label}
      <Icon name="chevron-down" size={13} strokeWidth={2.4} className="text-white/70" />
    </button>
  );
}

function Menu({ items, onPick }: { items: Item[]; onPick: () => void }) {
  return (
    <div className="absolute right-0 top-[calc(100%+6px)] z-20 flex min-w-[232px] flex-col gap-px rounded-[9px] border border-line bg-panel p-1.5">
      {items.map((it, i) =>
        "divider" in it ? (
          <div key={i} className="mx-2 my-1 h-px bg-line" />
        ) : "header" in it ? (
          <div key={i} className="px-2.5 pb-1 pt-2 text-[10px] font-bold uppercase tracking-wider text-txt-mute">
            {it.header}
          </div>
        ) : (
          <button
            key={i}
            onClick={() => { it.onClick(); onPick(); }}
            className={cn(
              "flex h-9 items-center gap-2.5 rounded-md px-2.5 text-left text-[13px] font-medium",
              it.danger ? "text-red-600 hover:bg-red-50" : "text-txt hover:bg-panel-2",
            )}
          >
            <Icon name={it.icon} size={16} strokeWidth={2} className={it.danger ? "text-red-500" : "text-txt-dim"} />
            {it.label}
          </button>
        ),
      )}
    </div>
  );
}

export function SelectionBar({ total }: { total: number }) {
  const { selection, selectAllMatching, selectAllFiltered, clearSelection } = useAudio();
  const { data: datasets = [] } = useDatasetsQuery();
  const [menu, setMenu] = useState<MenuName | null>(null);
  const ids = Object.keys(selection);
  const selCount = selectAllMatching ? total : ids.length;
  const toggle = (name: MenuName) => setMenu((m) => (m === name ? null : name));
  const count = selCount;

  const processItems: Item[] = [
    { header: "Processing jobs" },
    { label: "Split into segments", icon: "scissors", onClick: () => splitAction(count, datasets) },
    { label: "Transcribe", icon: "file-audio", onClick: () => transcribeAction(count) },
    { label: "Phonemize", icon: "sliders", onClick: () => phonemizeAction(count) },
    { label: "Normalize loudness", icon: "audio-lines", onClick: () => normalizeAction(count) },
    { label: "Denoise", icon: "wand", onClick: () => denoiseAction(count) },
    { divider: true },
    { label: "Calculate statistics", icon: "bar-chart", onClick: () => calculateStatisticsAction(count) },
  ];
  const datasetItems: Item[] = [
    { header: "Files & datasets" },
    { label: "Add to dataset", icon: "database", onClick: () => addDatasetAction(count, datasets) },
    { label: "Remove from dataset", icon: "database", onClick: () => removeDatasetAction(count, datasets) },
    { label: "Assign voice to segments", icon: "mic", onClick: () => assignVoiceAction(count) },
    { divider: true },
    { label: "Remove all segments", icon: "trash", danger: true, onClick: () => removeSegmentsAction(count) },
  ];

  return (
    <div className="relative mb-3 flex items-center gap-2.5 rounded-[9px] bg-gray-900 py-2.5 pl-4 pr-3">
      <div className="flex items-center gap-2.5">
        <span className="text-[13px] font-bold text-white">{selCount.toLocaleString()} selected</span>
      </div>
      {!selectAllMatching ? (
        <button onClick={selectAllFiltered} className="text-xs font-semibold text-blue-300">
          · Select all {total.toLocaleString()} matching filter
        </button>
      ) : null}
      <div className="flex-1" />
      <div className="relative">
        <DarkButton label="Process" icon="bolt" open={menu === "process"} onClick={() => toggle("process")} />
        {menu === "process" ? <Menu items={processItems} onPick={() => setMenu(null)} /> : null}
      </div>
      <div className="relative">
        <DarkButton label="Dataset & voice" icon="database" open={menu === "dataset"} onClick={() => toggle("dataset")} />
        {menu === "dataset" ? <Menu items={datasetItems} onPick={() => setMenu(null)} /> : null}
      </div>
      <button
        onClick={() => deleteFilesAction(count)}
        className="flex h-[34px] items-center gap-1.5 rounded-md bg-red-500 px-3 text-[12.5px] font-semibold text-white hover:bg-red-600"
      >
        <Icon name="trash" size={15} strokeWidth={2.2} />
        Delete
      </button>
      <button
        onClick={clearSelection}
        title="Clear selection"
        className="flex h-[34px] w-[34px] items-center justify-center rounded-md bg-white/10 text-white hover:bg-white/15"
      >
        <Icon name="x" size={16} strokeWidth={2.4} />
      </button>
    </div>
  );
}
