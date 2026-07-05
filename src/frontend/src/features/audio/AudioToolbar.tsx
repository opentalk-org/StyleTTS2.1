import { useDatasetsQuery } from "@/features/datasets/query";
import { Button } from "@/shared/ui/Button";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { uploadAction } from "./actions";
import { datasetOptions, sortOptions } from "./logic";
import { type AudioSort, useAudio } from "./store";

export function AudioToolbar() {
  const { query, dataset, sort, limit, setFilters } = useAudio();
  const { data: datasets = [] } = useDatasetsQuery();

  return (
    <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
      <SearchInput
        value={query}
        onChange={(v) => setFilters({ query: v, offset: 0 })}
        placeholder="Search files or speakers…"
      />
      <Select variant="mini" value={dataset} onChange={(v) => setFilters({ dataset: v, offset: 0 })} options={datasetOptions(datasets)} />
      <div className="flex-1" />
      <Select variant="mini" value={sort} onChange={(v) => setFilters({ sort: v as AudioSort, offset: 0 })} options={sortOptions()} />
      <Select
        variant="mini"
        value={String(limit)}
        onChange={(v) => setFilters({ limit: Number(v), offset: 0 })}
        options={[
          { value: "50", label: "50 per page" },
          { value: "100", label: "100 per page" },
          { value: "200", label: "200 per page" },
        ]}
      />
      <Button variant="primary" icon="upload" onClick={() => uploadAction(datasets)}>
        Upload
      </Button>
    </div>
  );
}
