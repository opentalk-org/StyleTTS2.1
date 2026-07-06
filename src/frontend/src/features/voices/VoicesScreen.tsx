import { Pager } from "@/shared/data/Pager";
import { VirtualTable } from "@/shared/data/VirtualTable";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { VoiceRow } from "./VoiceRow";
import { VoiceSkeleton } from "./VoiceSkeleton";
import { useVoiceActions, useVoicesQuery } from "./query";
import { useVoiceFilters } from "./store";

export function VoicesScreen() {
  const { query, limit, offset, set } = useVoiceFilters();
  const { data, isLoading, isError, refetch } = useVoicesQuery({ query, limit, offset });
  const { add } = useVoiceActions();

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const page = Math.floor(offset / limit);
  const pages = Math.max(1, Math.ceil(total / limit));
  const visibleEnd = Math.min(offset + rows.length, total);

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
