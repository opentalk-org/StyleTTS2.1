import { useState } from "react";

import { useNav } from "@/app/navStore";
import { openParamModal } from "@/shared/feedback/ParamModal";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { Input } from "@/shared/ui/Input";
import { DatasetCard } from "./DatasetCard";
import { useDatasetActions, useDatasetsQuery } from "./query";

export function DatasetsScreen() {
  const { data: datasets = [], isLoading, isError, refetch } = useDatasetsQuery();
  const { create, remove } = useDatasetActions();
  const go = useNav((s) => s.go);
  const [name, setName] = useState("");

  const submit = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    create(trimmed);
    setName("");
  };

  const importZip = () =>
    openParamModal({
      icon: "upload",
      title: "Import dataset ZIP",
      desc: "Drop a dataset archive to import files, segments, and metadata.",
      submitLabel: "Import",
      fields: [
        { type: "drop", label: "Drop a .zip archive or click to browse", hint: "Audio + segments.jsonl" },
        { key: "name", type: "text", label: "Dataset name", default: "imported_set" },
      ],
      onSubmit: (v) => showToast(`Importing into "${String(v.name)}"…`),
    });

  return (
    <div className="mx-auto max-w-[1000px] px-7 pb-16 pt-6">
      <div className="mb-5 flex gap-2.5">
        <Input
          className="h-10 max-w-[320px]"
          value={name}
          placeholder="New dataset name…"
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
        <Button variant="primary" size="lg" icon="plus" onClick={submit}>
          Create
        </Button>
        <div className="flex-1" />
        <Button variant="secondary" size="lg" icon="upload" onClick={importZip}>
          Import ZIP
        </Button>
      </div>
      {isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading datasets…</Card>
      ) : isError ? (
        <Card>
          <EmptyState
            icon="alert"
            title="Couldn't reach the backend"
            description="The datasets service didn't respond."
            action={
              <Button variant="primary" icon="refresh" onClick={() => refetch()}>
                Retry
              </Button>
            }
          />
        </Card>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3.5">
          {datasets.map((d) => (
            <DatasetCard key={d.id} dataset={d} onOpen={() => go("audio")} onDelete={remove} />
          ))}
        </div>
      )}
    </div>
  );
}
