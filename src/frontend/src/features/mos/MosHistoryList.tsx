import { useEffect, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SectionTitle } from "@/shared/ui/SectionTitle";
import type { MosRatingUpdateRequest } from "./api";
import { MosHistoryRow } from "./MosHistoryRow";
import { useMosHistoryMutations, useMosHistoryQuery } from "./query";

export function MosHistoryList({ datasetIds }: { datasetIds: string[] }) {
  const query = useMosHistoryQuery(datasetIds);
  const mutations = useMosHistoryMutations();
  const scrollRef = useRef<HTMLDivElement>(null);
  const items = query.data?.pages.flatMap((page) => page.rows) ?? [];
  const rows = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 92,
    overscan: 6,
  });
  const virtualRows = rows.getVirtualItems();

  useEffect(() => {
    const last = virtualRows.at(-1);
    if (!last || last.index < items.length - 5 || !query.hasNextPage || query.isFetchingNextPage) return;
    void query.fetchNextPage();
  }, [items.length, query, virtualRows]);

  const update = async (id: string, payload: MosRatingUpdateRequest) => {
    await mutations.update.mutateAsync({ id, payload });
  };
  const undo = async (id: string) => {
    await mutations.undo.mutateAsync(id);
  };
  const pending = mutations.update.isPending || mutations.undo.isPending;

  return (
    <div>
      <SectionTitle className="mb-3">Comparison history</SectionTitle>
      {query.isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading comparisons…</Card>
      ) : query.isError ? (
        <Card><EmptyState icon="alert" title="Could not load MOS history" /></Card>
      ) : !items.length ? (
        <Card><EmptyState icon="list-checks" title="No comparisons yet" description="Saved pair ratings will appear here." /></Card>
      ) : (
        <div ref={scrollRef} className="h-[420px] overflow-auto rounded-[10px] border border-line bg-panel-2 p-2">
          <div className="relative w-full" style={{ height: rows.getTotalSize() }}>
            {virtualRows.map((row) => {
              const item = items[row.index];
              if (!item) throw new Error(`MOS history row is unavailable: ${row.index}`);
              return (
                <div
                  key={item.id}
                  data-index={row.index}
                  ref={rows.measureElement}
                  className="absolute left-0 top-0 w-full pb-2"
                  style={{ transform: `translateY(${row.start}px)` }}
                >
                  <MosHistoryRow item={item} pending={pending} onUpdate={update} onUndo={undo} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
