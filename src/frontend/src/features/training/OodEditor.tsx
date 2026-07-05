import { useState } from "react";

import type { SchemaValues } from "@/shared/schema-form/types";
import { Icon } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import { Button } from "@/shared/ui/Button";
import { Select } from "@/shared/ui/Select";

import { FormSection } from "./FormSection";
import type { OodSetValue } from "./logic";

/** Out-of-domain reference text sets: list with per-row delete plus upload/add. */
export function OodEditor({
  values,
  availableSets,
  onChange,
}: {
  values: SchemaValues;
  availableSets: OodSetValue[];
  onChange: (values: SchemaValues) => void;
}) {
  const oodSets = values.sets as OodSetValue[];
  const choices = availableSets.filter((item) => !oodSets.some((selected) => selected.id === item.id));
  const [selectedId, setSelectedId] = useState("");
  const addOod = () => {
    const item = choices.find((choice) => choice.id === selectedId);
    if (!item) return;
    onChange({ ...values, sets: [...oodSets, item] });
    setSelectedId("");
  };
  const removeOod = (id: string) => {
    onChange({ ...values, sets: oodSets.filter((set) => set.id !== id) });
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
    </FormSection>
  );
}
