import { useEffect } from "react";

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

const HEAD = ["Wave", "File", "Dur"];

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
  const { query, language, dataset, sort, limit, cursor, page, selection, selectAllMatching, selectVisible, clearSelection, setVisibleIds, nextPage, previousPage } = useAudio();
  const { data, isFetching, isError, refetch } = useAudioFilesQuery({ query, language, dataset, sort, limit, cursor });
  const rows = data?.rows ?? [];
  const hasSelection = selectAllMatching || Object.keys(selection).length > 0;
  const visibleKey = rows.map((r) => r.id).join(",");
  const allSel = selectAllMatching || (rows.length > 0 && rows.every((r) => selection[r.id]));

  useEffect(() => {
    setVisibleIds(rows.map((r) => r.id));
  }, [visibleKey, setVisibleIds]);

  return (
    <div className="mx-auto flex h-full max-w-[1240px] flex-col px-7 pb-6 pt-5">
      <AudioToolbar />
      {hasSelection ? <SelectionBar /> : null}
      {isFetching ? (
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
            <span>{rows.length ? `Page ${page + 1} · ${rows.length} files` : "0 files"}</span>
            <Button variant="secondary" onClick={previousPage} disabled={page === 0}>Previous</Button>
            <Button
              variant="secondary"
              onClick={() => data?.next_cursor && nextPage(data.next_cursor)}
              disabled={!data?.has_more || !data.next_cursor}
            >
              Next
            </Button>
          </div>
          {rows.length ? (
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[10px] border border-line bg-panel">
              <Header allSel={allSel} onToggleAll={allSel ? clearSelection : selectVisible} />
              <div className="min-h-0 flex-1 overflow-y-auto">
                {rows.map((file, index) => <AudioRow key={file.id} file={file} index={page * limit + index} />)}
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
