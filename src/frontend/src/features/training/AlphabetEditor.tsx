import { useState } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Textarea } from "@/shared/ui/Textarea";
import { Button } from "@/shared/ui/Button";
import { Select } from "@/shared/ui/Select";

import { FormSection } from "./FormSection";
import { useTraining } from "./store";

const BASE_SYMBOLS = 178;

const PRESETS = [
  { value: "ipa", label: "IPA · default" },
  { value: "arpabet", label: "ARPAbet" },
  { value: "ipa-multi", label: "IPA · multilingual" },
  { value: "custom", label: "Custom" },
];

/** Phoneme alphabet editor: preset picker, live symbol count, and a base-count advisory. */
export function AlphabetEditor() {
  const alphabet = useTraining((s) => s.alphabet);
  const setAlphabet = useTraining((s) => s.setAlphabet);
  const [preset, setPreset] = useState("ipa");

  const count = alphabet.trim().split(/\s+/).filter(Boolean).length;
  const matches = count === BASE_SYMBOLS;

  return (
    <FormSection title="Phoneme alphabet" tag="Symbols">
      <div className="mb-3 flex flex-wrap items-center gap-2.5">
        <div className="min-w-[200px] flex-1">
          <Select value={preset} onChange={setPreset} options={PRESETS} />
        </div>
        <Button
          variant="ghost"
          icon="plus"
          onClick={() => showToast("Preset saved", preset)}
        >
          Save preset
        </Button>
        <div className="flex h-[38px] items-center gap-1.5 rounded-md bg-blue-50 px-3.5">
          <span className="text-[18px] font-extrabold tabular-nums text-blue-600">
            {count}
          </span>
          <span className="text-[11px] font-semibold text-blue-600">symbols</span>
        </div>
      </div>

      <Textarea
        value={alphabet}
        onChange={(e) => setAlphabet(e.target.value)}
        spellCheck={false}
        className="min-h-[76px] text-[15px] font-mono leading-[1.7]"
      />

      {matches ? (
        <div className="mt-2.5 flex items-center gap-2 text-xs font-semibold text-emerald-700">
          <Icon name="check-circle" size={15} strokeWidth={2.2} className="text-emerald-600" />
          Matches base checkpoint ({BASE_SYMBOLS} symbols).
        </div>
      ) : (
        <div className="mt-2.5 flex items-start gap-2 rounded-md bg-amber-50 px-3 py-2.5">
          <Icon name="alert" size={15} className="mt-px text-amber-600" />
          <span className="text-xs font-semibold leading-[1.45] text-amber-700">
            Symbol count ({count}) differs from base checkpoint ({BASE_SYMBOLS}).
            Embeddings for new symbols will be re-initialized.
          </span>
        </div>
      )}
    </FormSection>
  );
}
