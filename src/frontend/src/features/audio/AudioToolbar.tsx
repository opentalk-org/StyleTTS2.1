import { useDatasetsQuery } from "@/features/datasets/query";
import { Button } from "@/shared/ui/Button";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { uploadAction } from "./actions";
import { datasetOptions, sortOptions } from "./logic";
import { type AudioSort, useAudio } from "./store";

export function AudioToolbar() {
  const { query, dataset, sort, setFilters } = useAudio();
  const { data: datasets = [] } = useDatasetsQuery();

  return (
    <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
      <SearchInput
        value={query}
        onChange={(v) => setFilters({ query: v })}
        placeholder="Search files or speakers…"
      />
      <Select variant="mini" value={dataset} onChange={(v) => setFilters({ dataset: v })} options={datasetOptions(datasets)} />
      <div className="flex-1" />
      <Select variant="mini" value={sort} onChange={(v) => setFilters({ sort: v as AudioSort })} options={sortOptions()} />
      <Button variant="primary" icon="upload" onClick={() => uploadAction(datasets)}>
        Upload
      </Button>
    </div>
  );
}
