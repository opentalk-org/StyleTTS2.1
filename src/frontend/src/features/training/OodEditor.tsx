import type { SchemaValues } from "@/shared/schema-form/types";
import { Icon } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import { Button } from "@/shared/ui/Button";

import { FormSection } from "./FormSection";

type OodSetValue = { id: string; name: string; line_count: number };

/** Out-of-domain reference text sets: list with per-row delete plus upload/add. */
export function OodEditor({
  values,
  onChange,
}: {
  values: SchemaValues;
  onChange: (values: SchemaValues) => void;
}) {
  const oodSets = values.sets as OodSetValue[];
  const addOod = (name: string, lineCount: number) => {
    onChange({ ...values, sets: [...oodSets, { id: `ood_${Date.now()}`, name, line_count: lineCount }] });
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
        <button
          onClick={() =>
            addOod("uploaded_set.txt", 64 + Math.floor(Math.random() * 400))
          }
          className="flex h-[38px] flex-1 items-center justify-center gap-1.5 rounded-md border-2 border-dashed border-line-2 bg-panel text-xs font-semibold text-txt-dim cursor-pointer"
        >
          <Icon name="upload" size={14} />
          Upload .txt set
        </button>
        <Button
          variant="ghost"
          icon="plus"
          onClick={() => addOod("manual_prompts.txt", 24)}
        >
          Add set
        </Button>
      </div>
    </FormSection>
  );
}
