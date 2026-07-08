import { type ChangeEvent, useRef, useState } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import { Button } from "@/shared/ui/Button";
import { Select } from "@/shared/ui/Select";

import { FormSection } from "./FormSection";
import type { OodSetValue } from "./logic";

/** Out-of-domain reference text sets: select bucket files or upload a new one. */
export function OodEditor({
  selectedIds,
  availableSets,
  onChange,
  onCreate,
}: {
  selectedIds: string[];
  availableSets: OodSetValue[];
  onChange: (ids: string[]) => void;
  onCreate: (payload: { name: string; content: string }) => Promise<OodSetValue>;
}) {
  const oodSets = selectedIds.map((id) => availableSets.find((item) => item.id === id) ?? { id, name: id, line_count: 0 });
  const choices = availableSets.filter((item) => !selectedIds.includes(item.id));
  const [selectedId, setSelectedId] = useState("");
  const [creating, setCreating] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const addOod = () => {
    const item = choices.find((choice) => choice.id === selectedId);
    if (!item) return;
    onChange([...selectedIds, item.id]);
    setSelectedId("");
  };
  const removeOod = (id: string) => {
    onChange(selectedIds.filter((setId) => setId !== id));
  };
  const uploadOod = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setCreating(true);
    try {
      const content = (await file.text()).trim();
      if (!content) {
        showToast("That file is empty", undefined, "error");
        return;
      }
      const name = file.name.replace(/\.[^.]+$/, "").trim() || file.name;
      const item = await onCreate({ name, content });
      onChange([...selectedIds, item.id]);
    } catch {
      showToast("Could not upload OOD text file", undefined, "error");
    } finally {
      setCreating(false);
    }
  };

  return (
    <FormSection title="OOD reference texts" tag="Required">
      <p className="-mt-2 mb-3.5 text-xs leading-relaxed text-txt-mute">
        Out-of-domain prompts used to evaluate prosody generalization during
        training. At least one set required.
      </p>

      <div className="mb-3 flex flex-col gap-2">
        {oodSets.length === 0 ? (
          <div className="rounded-lg bg-panel-2 p-[18px] text-center text-xs text-txt-mute">
            No reference sets yet — add one below.
          </div>
        ) : (
          oodSets.map((s) => (
            <div
              key={s.id}
              className="flex items-center gap-2.5 rounded-md bg-panel-2 px-3 py-2.5"
            >
              <Icon name="list-checks" size={16} className="text-txt-dim" />
              <span className="flex-1 font-mono text-[13px] font-semibold text-txt">
                {s.name}
              </span>
              <span className="text-xs tabular-nums text-txt-mute">
                {s.line_count} lines
              </span>
              <IconButton
                icon="trash"
                size={28}
                iconSize={15}
                danger
                onClick={() => removeOod(s.id)}
              />
            </div>
          ))
        )}
      </div>

      <div className="flex gap-2">
        <div className="flex-1">
          <Select
            value={selectedId}
            onChange={setSelectedId}
            options={[
              { value: "", label: choices.length ? "— select OOD text file —" : "No OOD text files available" },
              ...choices.map((item) => ({ value: item.id, label: item.name })),
            ]}
          />
        </div>
        <Button
          variant="ghost"
          icon="plus"
          disabled={!selectedId}
          onClick={addOod}
        >
          Add set
        </Button>
      </div>

      <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-line bg-panel px-3 py-3">
        <p className="text-xs text-txt-mute">Upload a text file — one prompt per line.</p>
        <input ref={fileRef} type="file" accept=".txt,text/plain" className="hidden" onChange={uploadOod} />
        <Button
          variant="secondary"
          icon="upload"
          disabled={creating}
          onClick={() => fileRef.current?.click()}
        >
          {creating ? "Uploading…" : "Upload file"}
        </Button>
      </div>
    </FormSection>
  );
}
