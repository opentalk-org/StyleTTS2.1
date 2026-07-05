import { useDatasets } from "@/features/datasets/store";
import { VirtualTable } from "@/shared/data/VirtualTable";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import type { VoiceSort } from "./api";
import { VoiceRow } from "./VoiceRow";
import { VoiceSkeleton } from "./VoiceSkeleton";
import { useVoiceActions, useVoicesQuery } from "./query";
import { useVoiceFilters } from "./store";

export function VoicesScreen() {
  const { query, dataset, minSegments, sort, set } = useVoiceFilters();
  const { data, isLoading, isError, refetch } = useVoicesQuery({ query, dataset, minSegments, sort });
  const { add } = useVoiceActions();
  const datasets = useDatasets((s) => s.datasets);

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const datasetOptions = [
    { value: "all", label: "All datasets" },
    ...datasets.map((d) => ({ value: d.id, label: d.name })),
  ];

  return (
    <div className="mx-auto flex h-full max-w-[960px] flex-col px-7 pb-4 pt-5">
      <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
        <Button variant="primary" icon="plus" onClick={add} disabled={isLoading || isError}>
          New voice
        </Button>
        <SearchInput
          value={query}
          onChange={(v) => set({ query: v })}
          placeholder={total ? `Search ${total.toLocaleString()} voices…` : "Search voices…"}
        />
        <Select variant="mini" value={dataset} onChange={(v) => set({ dataset: v })} options={datasetOptions} />
        <Select
          variant="mini"
          value={String(minSegments)}
          onChange={(v) => set({ minSegments: Number(v) })}
          options={[
            { value: "0", label: "Any size" },
            { value: "1", label: "Has segments" },
            { value: "50", label: "≥ 50 segments" },
            { value: "200", label: "≥ 200 segments" },
          ]}
        />
        <Select
          variant="mini"
          value={sort}
          onChange={(v) => set({ sort: v as VoiceSort })}
          options={[
            { value: "name", label: "Sort: Name" },
            { value: "segments", label: "Sort: Most segments" },
            { value: "segments_asc", label: "Sort: Fewest segments" },
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
          <div className="mb-2.5 text-xs tabular-nums text-txt-mute">
            {rows.length.toLocaleString()} of {total.toLocaleString()} voices
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
