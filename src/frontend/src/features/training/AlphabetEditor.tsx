import { useState } from "react";

import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { Textarea } from "@/shared/ui/Textarea";
import { Select } from "@/shared/ui/Select";

import { FormSection } from "./FormSection";
import type { SchemaValues } from "@/shared/schema-form/types";
import type { TrainingConfig } from "./api";

/** Phoneme alphabet editor: preset picker, live symbol count, and a base-count advisory. */
export function AlphabetEditor({
  values,
  presets,
  baseSymbolCount,
  onChange,
  onSave,
}: {
  values: SchemaValues;
  presets: TrainingConfig[];
  baseSymbolCount: number | null;
  onChange: (values: SchemaValues) => void;
  onSave: (name: string, symbols: string) => void;
}) {
  const alphabet = String(values.symbols);
  const [preset, setPreset] = useState(String(values.preset));
  const [saveName, setSaveName] = useState("");

  const count = alphabet.trim().split(/\s+/).filter(Boolean).length;
  const matches = baseSymbolCount !== null && count === baseSymbolCount;
  const selectedPreset = presets.find((item) => item.id === preset || item.metadata.preset === preset);
  const options = [
    ...presets.map((item) => ({ value: item.id, label: item.name })),
    { value: "custom", label: "Custom" },
  ];

  return (
    <FormSection title="Phoneme alphabet" tag="Symbols">
      <div className="mb-3 grid gap-2.5 lg:grid-cols-[minmax(180px,1fr)_auto_auto] lg:items-center">
        <div className="min-w-[200px] flex-1">
          <Select
            value={selectedPreset?.id ?? "custom"}
            onChange={(value) => {
              const selected = presets.find((item) => item.id === value);
              if (!selected) {
                setPreset("custom");
                onChange({ ...values, preset: "custom" });
                return;
              }
              setPreset(String(selected.metadata.preset ?? "custom"));
              onChange({
                ...values,
                preset: String(selected.metadata.preset ?? "custom"),
                symbols: alphabetSymbols(selected),
              });
            }}
            options={options}
          />
        </div>
        <div className="flex h-[38px] items-center gap-1.5 rounded-md bg-blue-50 px-3.5">
          <span className="text-[18px] font-extrabold tabular-nums text-blue-600">
            {count}
          </span>
          <span className="text-[11px] font-semibold text-blue-600">symbols</span>
        </div>
        <div className="flex min-w-0 gap-2">
          <Input filled value={saveName} onChange={(event) => setSaveName(event.target.value)} placeholder="New alphabet name" />
          <Button
            size="sm"
            icon="download"
            disabled={!saveName.trim() || !alphabet.trim()}
            onClick={() => {
              onSave(saveName.trim(), alphabet);
              setSaveName("");
            }}
          >
            Save
          </Button>
        </div>
      </div>

      <Textarea
        value={alphabet}
        onChange={(event) => onChange({ ...values, symbols: event.target.value })}
        spellCheck={false}
        className="min-h-[76px] text-[15px] font-mono leading-[1.7]"
      />

      {baseSymbolCount === null ? (
        <div className="mt-2.5 text-xs font-semibold text-txt-mute">
          Select a checkpoint with symbol metadata to compare embedding size.
        </div>
      ) : matches ? (
        <div className="mt-2.5 flex items-center gap-2 text-xs font-semibold text-emerald-700">
          <Icon name="check-circle" size={15} strokeWidth={2.2} className="text-emerald-600" />
          Matches selected checkpoint ({baseSymbolCount} symbols).
        </div>
      ) : (
        <div className="mt-2.5 flex items-start gap-2 rounded-md bg-amber-50 px-3 py-2.5">
          <Icon name="alert" size={15} className="mt-px text-amber-600" />
          <span className="text-xs font-semibold leading-[1.45] text-amber-700">
            Symbol count ({count}) differs from selected checkpoint ({baseSymbolCount}).
            Embeddings for new symbols will be re-initialized.
          </span>
        </div>
      )}
    </FormSection>
  );
}

function alphabetSymbols(config: TrainingConfig): string {
  const raw = config.metadata.symbols;
  if (Array.isArray(raw)) return raw.map(String).join(" ");
  return String(raw ?? "");
}
