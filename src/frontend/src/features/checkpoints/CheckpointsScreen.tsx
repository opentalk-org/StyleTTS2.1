import { useState } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { IconButton } from "@/shared/ui/IconButton";
import { SearchInput } from "@/shared/ui/SearchInput";
import { SectionTitle } from "@/shared/ui/SectionTitle";
import { Select } from "@/shared/ui/Select";
import { CheckpointRow } from "./CheckpointRow";
import { CATALOG, groupCheckpoints } from "./logic";
import { useCheckpointsQuery } from "./query";

export function CheckpointsScreen() {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("all");
  const checkpoints = useCheckpointsQuery();
  const groups = groupCheckpoints(checkpoints.data ?? [], query, type);

  return (
    <div className="mx-auto max-w-[1080px] px-7 pb-16 pt-5">
      <div className="mb-4 flex flex-wrap items-center gap-2.5">
        <SearchInput value={query} onChange={setQuery} placeholder="Search checkpoints..." />
        <Select
          variant="mini"
          value={type}
          onChange={setType}
          options={[
            { value: "all", label: "All types" },
            { value: "styletts2", label: "StyleTTS2" },
            { value: "asr", label: "ASR" },
            { value: "f0", label: "F0" },
            { value: "plbert", label: "PL-BERT" },
          ]}
        />
      </div>

      {checkpoints.isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading checkpoints...</Card>
      ) : checkpoints.isError ? (
        <Card>
          <EmptyState icon="alert" title="Couldn't reach the backend" description="The checkpoints service didn't respond." />
        </Card>
      ) : Object.keys(groups).length ? (
        <div className="flex flex-col gap-[18px]">
          {Object.entries(groups).map(([job, items]) => (
            <div key={job}>
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-txt-dim">
                <Icon name="activity" size={14} strokeWidth={2} className="text-txt-mute" />
                {job === "-" ? "Imported / uploaded" : "From job"}
                {job !== "-" ? <span className="font-mono text-blue-600">{job}</span> : null}
              </div>
              <Card className="overflow-hidden">
                <div className="overflow-x-auto">
                  <div className="min-w-[640px]">
                    {items.map((checkpoint) => <CheckpointRow key={checkpoint.id} checkpoint={checkpoint} />)}
                  </div>
                </div>
              </Card>
            </div>
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState icon="box" title="No checkpoints match your filters." />
        </Card>
      )}

      <div className="mt-6">
        <SectionTitle className="mb-3">Pretrained catalog</SectionTitle>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3">
          {CATALOG.map((item) => (
            <Card key={item.file} className="flex items-center gap-3 p-4">
              <div className="flex h-[38px] w-[38px] flex-none items-center justify-center rounded-[9px] bg-blue-50 text-blue-600">
                <Icon name="box" size={18} strokeWidth={2.2} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-bold text-txt">{item.name}</div>
                <div className="font-mono text-[11px] text-txt-mute">
                  {item.file} · {item.size}
                </div>
              </div>
              <IconButton icon="download" title="Download" onClick={() => showToast("Download queued", item.file)} />
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
