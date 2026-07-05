import { Field } from "@/shared/ui/form/Field";
import { Input } from "@/shared/ui/Input";

import { AlphabetEditor } from "./AlphabetEditor";
import { FormSection } from "./FormSection";
import { FormSelect, opts } from "./FormSelect";

const DATASETS = ["vox_studio_v3", "narration_set", "podcast_clean", "librispeech_360"];

/** Compact finetune form for the F0 pitch extractor and ASR aligner models. */
export function SmallModelForm({ variant }: { variant: "f0" | "asr" }) {
  const isAsr = variant === "asr";
  const pretrained = opts([
    { value: "", label: "— train from scratch —" },
    isAsr ? "asr_base.pth" : "f0_base.pth",
  ]);

  return (
    <>
      <FormSection
        title={isAsr ? "ASR model" : "F0 model"}
        tag={isAsr ? "Aligner" : "Pitch extractor"}
      >
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Display name">
            <Input filled defaultValue={isAsr ? "asr_v2" : "f0_v2"} />
          </Field>
          <Field label="Training dataset">
            <FormSelect defaultValue="vox_studio_v3" options={opts(DATASETS)} />
          </Field>
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Pretrained (optional)">
            <FormSelect defaultValue="" options={pretrained} />
          </Field>
          <Field label="Validation samples">
            <Input filled type="number" defaultValue={32} min={0} />
          </Field>
        </div>
      </FormSection>

      <FormSection title="Optimization" tag="Optimizer">
        <div className={isAsr ? "grid grid-cols-4 gap-3.5" : "grid grid-cols-3 gap-3.5"}>
          <Field label="Batch size">
            <Input filled type="number" defaultValue={32} />
          </Field>
          <Field label="Learning rate">
            <Input filled defaultValue="5e-4" />
          </Field>
          <Field label="Epochs">
            <Input filled type="number" defaultValue={100} />
          </Field>
          {isAsr ? (
            <Field label="Dataloader workers">
              <Input filled type="number" defaultValue={8} min={0} />
            </Field>
          ) : null}
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Save interval (epochs)">
            <Input filled type="number" defaultValue={10} />
          </Field>
          <div />
        </div>
      </FormSection>

      {isAsr ? <AlphabetEditor /> : null}
    </>
  );
}
