import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { fmtAgo } from "@/shared/format";
import { Icon } from "@/shared/icons";
import type { StatisticsSummary } from "./api";

export function StatisticsEntryPicker({
  entries,
  value,
  onChange,
}: {
  entries: StatisticsSummary[];
  value: string | null;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const selected = entries.find((entry) => entry.id === value) ?? null;
  const filtered = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return [...entries]
      .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at))
      .filter((entry) => !needle || entry.name.toLocaleLowerCase().includes(needle));
  }, [entries, query]);
  const rows = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 66,
    overscan: 7,
  });

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", closeOutside);
    return () => window.removeEventListener("pointerdown", closeOutside);
  }, [open]);

  return (
    <div ref={rootRef} className="relative min-w-[360px] max-w-[540px] flex-1">
      <button
        type="button"
        className="flex h-12 w-full cursor-pointer items-center gap-3 rounded-lg border border-line bg-panel px-3 text-left shadow-sm transition-colors hover:border-line-2 hover:bg-panel-2"
        onClick={() => setOpen((current) => !current)}
      >
        <span className="grid h-8 w-8 flex-none place-items-center rounded-md bg-blue-50 text-blue-600">
          <Icon name="bar-chart" size={15} strokeWidth={2.2} />
        </span>
        <span className="min-w-0 flex-1">
          <strong className="block truncate text-[13px] text-txt">{selected?.name ?? "Select statistics entry"}</strong>
          <span className="block truncate text-[10.5px] text-txt-mute">
            {selected ? `${selected.file_count.toLocaleString()} files · ${formatCreatedAt(selected.created_at)} · ${fmtAgo(Date.parse(selected.created_at))}` : `${entries.length} entries`}
          </span>
        </span>
        <Icon name="chevron-down" size={14} className="flex-none text-txt-mute" />
      </button>

      {open ? (
        <div className="absolute left-0 top-full z-40 mt-1.5 w-full min-w-[440px] overflow-hidden rounded-lg border border-line bg-panel shadow-[0_18px_55px_rgba(17,24,39,0.22)]">
          <div className="border-b border-line p-2">
            <label className="flex h-9 items-center gap-2 rounded-md border border-line bg-panel-2 px-2.5 focus-within:border-blue-500">
              <Icon name="search" size={14} className="text-txt-mute" />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") setOpen(false);
                }}
                placeholder="Search statistics entries…"
                className="min-w-0 flex-1 border-0 bg-transparent text-[12.5px] text-txt outline-none placeholder:text-txt-mute"
              />
              <span className="font-mono text-[10px] text-txt-mute">{filtered.length}</span>
            </label>
          </div>
          {filtered.length ? (
            <div ref={scrollRef} className="h-[min(330px,50vh)] overflow-auto">
              <div className="relative w-full" style={{ height: rows.getTotalSize() }}>
                {rows.getVirtualItems().map((row) => {
                  const entry = filtered[row.index];
                  if (!entry) throw new Error(`Statistics entry row is unavailable: ${row.index}`);
                  const active = entry.id === value;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      className={`absolute left-0 top-0 flex w-full cursor-pointer items-center gap-3 border-b border-line px-3 text-left hover:bg-blue-50 ${active ? "bg-blue-50" : "bg-panel"}`}
                      style={{ height: row.size, transform: `translateY(${row.start}px)` }}
                      onClick={() => {
                        onChange(entry.id);
                        setOpen(false);
                        setQuery("");
                      }}
                    >
                      <span className={`h-2 w-2 flex-none rounded-full ${active ? "bg-blue-500" : "bg-slate-300"}`} />
                      <span className="min-w-0 flex-1">
                        <strong className="block truncate text-[12.5px] text-txt">{entry.name}</strong>
                        <span className="block truncate text-[10.5px] text-txt-mute">Created {formatCreatedAt(entry.created_at)} · {fmtAgo(Date.parse(entry.created_at))}</span>
                      </span>
                      <span className="flex-none rounded bg-panel-2 px-2 py-1 font-mono text-[10px] tabular-nums text-txt-dim">{entry.file_count.toLocaleString()} files</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="px-4 py-8 text-center text-[12px] text-txt-mute">No matching statistics entries.</div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function formatCreatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}
