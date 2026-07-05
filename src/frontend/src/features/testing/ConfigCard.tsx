import { useCheckpoints } from "@/features/checkpoints/store";
import { seedStyleRefs } from "@/mock/data";
import { showToast } from "@/shared/feedback/Toast";
import { Card } from "@/shared/ui/Card";
import { Field } from "@/shared/ui/form/Field";
import { Slider } from "@/shared/ui/form/Slider";
import { Select, type Option } from "@/shared/ui/Select";
import { Textarea } from "@/shared/ui/Textarea";
import { checkpointOptions } from "./logic";
import { useTesting } from "./store";

const LANGS: Option[] = [
  { value: "en-us", label: "English (US)" },
  { value: "en-gb", label: "English (UK)" },
  { value: "es", label: "Spanish" },
  { value: "de", label: "German" },
];

const WEIGHTS: Option[] = [
  { value: "best.pth", label: "best.pth" },
  { value: "ep50.pth", label: "epoch_50.pth" },
  { value: "ep35.pth", label: "epoch_35.pth" },
];

const STYLE_REFS: Option[] = [
  ...seedStyleRefs().map((r) => ({ value: r.id, label: `${r.name}  (${r.voice})` })),
  { value: "upload", label: "⤴ Upload reference…" },
];

function GroupTitle({ children }: { children: string }) {
  return (
    <div className="text-[11px] font-bold uppercase tracking-[0.06em] text-blue-500">
      {children}
    </div>
  );
}

/** Auto-fit grid of form fields; each cell can shrink below content width. */
function FieldGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-x-5 gap-y-4 [&>*]:min-w-0">
      {children}
    </div>
  );
}

/** Single-synthesis config: Text & inference / Model / Style reference. */
export function ConfigCard() {
  const checkpoints = useCheckpoints((s) => s.checkpoints);
  const c = useTesting((s) => s.single);
  const setSingle = useTesting((s) => s.setSingle);

  const onStyleRef = (v: string) => {
    if (v === "upload") {
      showToast("Choose a .wav style reference");
      return;
    }
    setSingle("styleRef", v);
  };

  return (
    <Card className="flex flex-col gap-5 rounded-xl px-6 py-[22px]">
      <div className="flex flex-col gap-2.5">
        <GroupTitle>Text &amp; inference</GroupTitle>
        <Field label="Text">
          <Textarea value={c.text} onChange={(e) => setSingle("text", e.target.value)} />
        </Field>
        <FieldGrid>
          <Field label="Language">
            <Select value={c.lang} onChange={(v) => setSingle("lang", v)} options={LANGS} />
          </Field>
          <Field label="Diffusion steps">
            <Slider value={c.steps} onChange={(v) => setSingle("steps", v)} min={1} max={20} step={1} />
          </Field>
          <Field label="Embedding scale">
            <Slider
              value={c.emb}
              onChange={(v) => setSingle("emb", v)}
              min={0.5}
              max={3}
              step={0.1}
              format={(v) => v.toFixed(1)}
            />
          </Field>
        </FieldGrid>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Model</GroupTitle>
        <FieldGrid>
          <Field label="Checkpoint">
            <Select
              value={c.ckpt}
              onChange={(v) => setSingle("ckpt", v)}
              options={checkpointOptions(checkpoints)}
            />
          </Field>
          <Field label="Weights file">
            <Select value={c.weights} onChange={(v) => setSingle("weights", v)} options={WEIGHTS} />
          </Field>
        </FieldGrid>
      </div>

      <div className="h-px bg-line" />

      <div className="flex flex-col gap-2.5">
        <GroupTitle>Style reference</GroupTitle>
        <FieldGrid>
          <Field label="Reference">
            <Select value={c.styleRef} onChange={onStyleRef} options={STYLE_REFS} />
          </Field>
          <Field label="Style mix" hint="Timbre adherence to reference.">
            <Slider
              value={c.styleMix}
              onChange={(v) => setSingle("styleMix", v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
          </Field>
          <Field label="Prosody mix" hint="Rhythm & intonation from reference.">
            <Slider
              value={c.prosodyMix}
              onChange={(v) => setSingle("prosodyMix", v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
            />
          </Field>
        </FieldGrid>
      </div>
    </Card>
  );
}
