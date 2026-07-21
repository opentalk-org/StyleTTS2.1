import { useEffect } from "react";

import { Pager } from "@/shared/data/Pager";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { cn } from "@/shared/ui/cn";
import { EmptyState } from "@/shared/ui/EmptyState";
import { AUDIO_COLS, AudioRow } from "./AudioRow";
import { AudioToolbar } from "./AudioToolbar";
import { SelectionBar } from "./SelectionBar";
import { useAudioFilesQuery } from "./query";
import { useAudio } from "./store";

const HEAD = ["Wave", "File", "Speaker", "Dur"];

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
      <span />
    </div>
  );
}

export function AudioScreen() {
  const { query, dataset, sort, limit, offset, selection, selectAllMatching, selectVisible, clearSelection, setVisibleIds, setFilters } = useAudio();
  const { data, isLoading, isError, refetch } = useAudioFilesQuery({ query, dataset, sort, limit, offset });
  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  const visibleEnd = Math.min(offset + rows.length, total);
  const hasSelection = selectAllMatching || Object.keys(selection).length > 0;
  const visibleKey = rows.map((r) => r.id).join(",");
  const allSel = selectAllMatching || (rows.length > 0 && rows.every((r) => selection[r.id]));

  useEffect(() => {
    setVisibleIds(rows.map((r) => r.id));
  }, [visibleKey, setVisibleIds]);

  return (
    <div className="mx-auto flex h-full max-w-[1240px] flex-col px-7 pb-6 pt-5">
      <AudioToolbar />
      {hasSelection ? <SelectionBar total={total} /> : null}
      {isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading audio files...</Card>
      ) : isError ? (
        <Card>
          <EmptyState
            icon="alert"
            title="Couldn't reach the backend"
            description="The audio file service didn't respond."
            action={
              <Button variant="primary" icon="refresh" onClick={() => refetch()}>
                Retry
              </Button>
            }
          />
        </Card>
      ) : (
        <>
          <div className="mb-2.5 flex items-center gap-3 text-xs tabular-nums text-txt-mute">
            <span>
              {total ? `${(offset + 1).toLocaleString()}-${visibleEnd.toLocaleString()}` : "0"} of {total.toLocaleString()} files
            </span>
            <Pager page={page} pages={pages} onChange={(next) => setFilters({ offset: next * limit })} />
          </div>
          {rows.length ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[10px] border border-line bg-panel">
              <Header allSel={allSel} onToggleAll={allSel ? clearSelection : selectVisible} />
              <div className="min-h-0 flex-1 overflow-y-auto">
                {rows.map((file, index) => <AudioRow key={file.id} file={file} index={offset + index} />)}
              </div>
            </div>
          ) : (
            <Card>
              <EmptyState icon="file-audio" title="No audio files match your filters." />
            </Card>
          )}
        </>
      )}
    </div>
  );
}
