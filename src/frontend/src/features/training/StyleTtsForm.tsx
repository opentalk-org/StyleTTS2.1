import { Field } from "@/shared/ui/form/Field";
import { NumberInput } from "@/shared/ui/form/NumberInput";
import { Toggle } from "@/shared/ui/form/Toggle";
import { Input } from "@/shared/ui/Input";

import { AlphabetEditor } from "./AlphabetEditor";
import { AssetSlot } from "./AssetSlot";
import { FormSection } from "./FormSection";
import { FormSelect, opts } from "./FormSelect";
import { OodEditor } from "./OodEditor";
import { useTraining } from "./store";
import type { Toggles } from "./store";

const DATASETS = ["vox_studio_v3", "narration_set", "podcast_clean", "librispeech_360"];

const CHECKPOINTS = opts([
  { value: "", label: "— select base checkpoint —" },
  "styletts2_libritts.pth",
  "styletts2_ljspeech.pth",
  "vox_studio_v2_ep50.pth",
]);

const TOGGLE_ROWS: { key: keyof Toggles; title: string; sub: string }[] = [
  { key: "multispeaker", title: "Multi-speaker mode", sub: "Per-speaker style encoder" },
  { key: "stagewise", title: "Checkpoint each stage", sub: "Save base / diffusion / joint separately" },
  { key: "mixedprec", title: "Mixed precision", sub: "Faster, slightly less stable" },
];

/** The full StyleTTS finetune configuration form. */
export function StyleTtsForm() {
  const seqSeconds = useTraining((s) => s.seqSeconds);
  const setSeqSeconds = useTraining((s) => s.setSeqSeconds);
  const toggles = useTraining((s) => s.toggles);
  const setToggle = useTraining((s) => s.setToggle);

  const frames = Math.round((seqSeconds * 24000) / 300);

  return (
    <>
      <FormSection title="Identity & data" tag="Run">
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Display name">
            <Input filled defaultValue="vox_studio_v3" placeholder="my-finetune-run" />
          </Field>
          <Field label="Training dataset">
            <FormSelect defaultValue="vox_studio_v3" options={opts(DATASETS)} />
          </Field>
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-2 gap-3.5">
          <Field label="Validation samples" hint="Held out from training each epoch.">
            <Input filled type="number" defaultValue={32} min={0} max={512} step={1} />
          </Field>
          <Field label="Base checkpoint" hint="Required — finetuning resumes from here.">
            <FormSelect defaultValue="" options={CHECKPOINTS} />
          </Field>
        </div>
      </FormSection>

      <FormSection title="Asset slots" tag="Optional pretrained">
        <div className="grid grid-cols-3 gap-3.5">
          <AssetSlot label="F0 model" file="jdc_f0.pth" />
          <AssetSlot label="ASR model" file="asr_aligner.pth" />
          <AssetSlot label="PL-BERT" file="step_1M.t7" />
        </div>
      </FormSection>

      <AlphabetEditor />

      <FormSection title="Optimization" tag="Optimizer">
        <div className="grid grid-cols-3 gap-3.5">
          <Field label="Batch size">
            <Input filled type="number" defaultValue={16} min={1} max={128} step={1} />
          </Field>
          <Field label="Learning rate">
            <Input filled defaultValue="1e-4" />
          </Field>
          <Field label="Numeric precision">
            <FormSelect defaultValue="bf16" options={opts(["fp32", "fp16", "bf16"])} />
          </Field>
        </div>
        <div className="mt-4 mb-2.5 text-xs font-semibold text-txt-dim">
          Gradient clipping (max norm)
        </div>
        <div className="grid grid-cols-3 gap-3.5">
          <Field label="Total">
            <Input filled defaultValue="5.0" />
          </Field>
          <Field label="Diffusion">
            <Input filled defaultValue="1.0" />
          </Field>
          <Field label="SLM">
            <Input filled defaultValue="0.5" />
          </Field>
        </div>
      </FormSection>

      <FormSection title="Schedule" tag="Epochs & sequence">
        <div className="grid grid-cols-3 gap-3.5">
          <Field label="Epochs · base">
            <Input filled type="number" defaultValue={30} min={0} />
          </Field>
          <Field label="Epochs · diffusion">
            <Input filled type="number" defaultValue={15} min={0} />
          </Field>
          <Field label="Epochs · joint">
            <Input filled type="number" defaultValue={5} min={0} />
          </Field>
        </div>
        <div className="h-3.5" />
        <div className="grid grid-cols-3 gap-3.5">
          <Field label="Max sequence (sec)" hint={`≈ ${frames} frames @ 300 hop`}>
            <NumberInput value={seqSeconds} onChange={setSeqSeconds} min={1} max={30} step={0.5} />
          </Field>
          <Field label="Save interval (epochs)">
            <Input filled type="number" defaultValue={5} min={1} />
          </Field>
          <Field label="Decoder">
            <FormSelect defaultValue="hifigan" options={opts(["hifigan", "istftnet"])} />
          </Field>
        </div>
      </FormSection>

      <FormSection title="SLM adversarial" tag="Discriminator">
        <div className="grid grid-cols-3 gap-3.5">
          <Field label="SLM weight">
            <Input filled defaultValue="0.2" />
          </Field>
          <Field label="Diffusion samples">
            <Input filled type="number" defaultValue={3} min={1} />
          </Field>
          <Field label="Scale">
            <Input filled defaultValue="0.01" />
          </Field>
        </div>
        <div className="mt-4 flex flex-col gap-3">
          {TOGGLE_ROWS.map((r) => (
            <label key={r.key} className="flex items-center gap-3 cursor-pointer">
              <span className="flex-1">
                <div className="text-[13px] font-semibold text-txt">{r.title}</div>
                <div className="text-xs text-txt-mute">{r.sub}</div>
              </span>
              <Toggle
                checked={toggles[r.key]}
                onChange={(v) => setToggle(r.key, v)}
              />
            </label>
          ))}
        </div>
      </FormSection>

      <OodEditor />
    </>
  );
}
