import { Field } from "@/shared/ui/form/Field";
import { Toggle } from "@/shared/ui/form/Toggle";
import { Select } from "@/shared/ui/Select";

import type {
  ProsodySource,
  TrainingStageSpec,
  ValidationDurationSource,
} from "./templates";

export function ValidationSettingsEditor({
  value,
  onChange,
}: {
  value: TrainingStageSpec["validation"];
  onChange: (value: TrainingStageSpec["validation"]) => void;
}) {
  const sources = [
    { value: "ground_truth", label: "Ground truth" },
    { value: "predicted", label: "Predicted" },
  ];
  return (
    <div className="mt-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-txt-mute">
        Validation synthesis
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="F0">
          <Select
            value={value.f0_source}
            options={sources}
            onChange={(f0_source) => onChange({
              ...value,
              f0_source: f0_source as ProsodySource,
            })}
          />
        </Field>
        <Field label="N / energy">
          <Select
            value={value.norm_source}
            options={sources}
            onChange={(norm_source) => onChange({
              ...value,
              norm_source: norm_source as ProsodySource,
            })}
          />
        </Field>
        <Field label="Duration">
          <Select
            value={value.duration_source}
            options={sources}
            onChange={(duration_source) => onChange({
              ...value,
              duration_source:
                duration_source as ValidationDurationSource,
            })}
          />
        </Field>
        <Field label="Diffusion style">
          <div className="flex h-10 items-center">
            <Toggle
              checked={value.diffusion}
              onChange={(diffusion) => onChange({ ...value, diffusion })}
            />
          </div>
        </Field>
      </div>
    </div>
  );
}
