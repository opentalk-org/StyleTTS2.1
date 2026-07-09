import { useEffect } from "react";

import { Pager } from "@/shared/data/Pager";
import { VirtualTable } from "@/shared/data/VirtualTable";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { cn } from "@/shared/ui/cn";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { VoiceRow } from "./VoiceRow";
import { VoiceSelectionBar } from "./VoiceSelectionBar";
import { VoiceSkeleton } from "./VoiceSkeleton";
import { useVoiceActions, useVoicesQuery } from "./query";
import { useVoiceFilters } from "./store";

export function VoicesScreen() {
  const {
    query,
    limit,
    offset,
    set,
    selection,
    selectAllMatching,
    setVisibleIds,
    selectVisible,
    clearSelection,
  } = useVoiceFilters();
  const { data, isLoading, isError, refetch } = useVoicesQuery({ query, limit, offset });
  const { add } = useVoiceActions();

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  const visibleEnd = Math.min(offset + rows.length, total);
  const hasSelection = selectAllMatching || Object.keys(selection).length > 0;
  const allSel = selectAllMatching || (rows.length > 0 && rows.every((r) => selection[r.id]));
  const visibleKey = rows.map((r) => r.id).join(",");

  useEffect(() => {
    setVisibleIds(rows.map((r) => r.id));
  }, [visibleKey, setVisibleIds]);

  return (
    <div className="mx-auto flex h-full max-w-[960px] flex-col px-7 pb-4 pt-5">
      <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
        <Button variant="primary" icon="plus" onClick={add} disabled={isLoading || isError}>
          New voice
        </Button>
        <SearchInput
          value={query}
          onChange={(v) => set({ query: v, offset: 0 })}
          placeholder={total ? `Search ${total.toLocaleString()} voices…` : "Search voices…"}
        />
        <Select
          variant="mini"
          value={String(limit)}
          onChange={(v) => set({ limit: Number(v), offset: 0 })}
          options={[
            { value: "50", label: "50 per page" },
            { value: "100", label: "100 per page" },
            { value: "200", label: "200 per page" },
          ]}
        />
      </div>

      {hasSelection ? <VoiceSelectionBar total={total} /> : null}

      {isLoading ? (
        <Card className="overflow-hidden">
          <VoiceSkeleton />
        </Card>
      ) : isError ? (
        <Card>
          <EmptyState
            icon="alert"
            title="Couldn't reach the backend"
            description="The voices service didn't respond."
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
              {total ? `${(offset + 1).toLocaleString()}-${visibleEnd.toLocaleString()}` : "0"} of{" "}
              {total.toLocaleString()} voices
            </span>
            <Pager page={page} pages={pages} onChange={(p) => set({ offset: p * limit })} />
          </div>
          {rows.length ? (
            <VirtualTable
              count={rows.length}
              estimateRowHeight={66}
              className="flex-1"
              header={
                <div className="mb-1 flex items-center gap-3 px-3.5">
                  <button
                    onClick={allSel ? clearSelection : selectVisible}
                    className="flex"
                    title={allSel ? "Clear selection" : "Select all on this page"}
                  >
                    <span
                      className={cn(
                        "flex h-[18px] w-[18px] items-center justify-center rounded",
                        allSel ? "bg-blue-500" : "border-2 border-line-2 bg-panel",
                      )}
                    >
                      {allSel ? <Icon name="check" size={12} strokeWidth={3} className="text-white" /> : null}
                    </span>
                  </button>
                  <span className="text-[11px] font-bold uppercase tracking-wider text-txt-mute">
                    Select all
                  </span>
                </div>
              }
              renderRow={(i) => <VoiceRow voice={rows[i]!} />}
            />
          ) : (
            <Card>
              <EmptyState icon="mic" title="No voices match your filters." />
            </Card>
          )}
        </>
      )}
    </div>
  );
}
