import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import type { Dataset } from "@/features/datasets/api";

export function MosDatasetPicker({
  datasets,
  selectedIds,
  onToggle,
}: {
  datasets: Dataset[];
  selectedIds: string[];
  onToggle: (datasetId: string) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const rows = useVirtualizer({
    count: datasets.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 38,
    overscan: 8,
  });

  if (!datasets.length) return <div className="text-[13px] text-txt-mute">No datasets available.</div>;

  return (
    <div ref={scrollRef} className="h-[190px] overflow-auto rounded-lg border border-line bg-panel-2">
      <div className="relative w-full" style={{ height: rows.getTotalSize() }}>
        {rows.getVirtualItems().map((row) => {
          const dataset = datasets[row.index];
          if (!dataset) throw new Error(`Dataset row is unavailable: ${row.index}`);
          return (
            <label
              key={dataset.id}
              className="absolute left-0 top-0 flex w-full cursor-pointer items-center gap-3 border-b border-line px-3 text-[13px] text-txt"
              style={{ height: row.size, transform: `translateY(${row.start}px)` }}
            >
              <input
                type="checkbox"
                checked={selectedIds.includes(dataset.id)}
                onChange={() => onToggle(dataset.id)}
                className="h-4 w-4 accent-blue-500"
              />
              <span className="min-w-0 flex-1 truncate font-semibold">{dataset.name}</span>
              <span className="font-mono text-[11px] tabular-nums text-txt-mute">{dataset.files.toLocaleString()}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
