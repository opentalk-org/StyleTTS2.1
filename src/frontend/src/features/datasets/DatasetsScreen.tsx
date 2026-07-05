import { useState } from "react";

import { useNav } from "@/app/navStore";
import { openParamModal } from "@/shared/feedback/ParamModal";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { DatasetCard } from "./DatasetCard";
import { useDatasets } from "./store";

export function DatasetsScreen() {
  const { datasets, create, remove } = useDatasets();
  const go = useNav((s) => s.go);
  const [name, setName] = useState("");

  const submit = () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    create(trimmed);
    setName("");
    showToast(`Dataset "${trimmed}" created`);
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
      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-3.5">
        {datasets.map((d) => (
          <DatasetCard key={d.id} dataset={d} onOpen={() => go("audio")} onDelete={remove} />
        ))}
      </div>
    </div>
  );
}
