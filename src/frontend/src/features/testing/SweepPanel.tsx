import { SPEAKERS } from "@/mock/constants";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { Field } from "@/shared/ui/form/Field";
import { Slider } from "@/shared/ui/form/Slider";
import { Textarea } from "@/shared/ui/Textarea";
import { cn } from "@/shared/ui/cn";
import { useTesting } from "./store";

function VoiceChip({ name, on, onClick }: { name: string; on: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex h-[34px] items-center gap-1.5 rounded-full border-[1.5px] px-3.5 text-[12.5px] font-semibold cursor-pointer transition-colors",
        on
          ? "border-blue-500 bg-blue-50 text-blue-700"
          : "border-line-2 bg-panel text-txt-dim",
      )}
    >
      {on ? <Icon name="check" size={13} strokeWidth={3} className="text-blue-600" /> : null}
      {name}
    </button>
  );
}

/** Sweep config form + a grid of per-voice result cards. */
export function SweepPanel() {
  const sweep = useTesting((s) => s.sweep);
  const results = useTesting((s) => s.sweepResults);
  const setSweep = useTesting((s) => s.setSweep);
  const toggleVoice = useTesting((s) => s.toggleVoice);
  const genSweep = useTesting((s) => s.genSweep);

  const selected = SPEAKERS.filter((n) => sweep.voices[n]);

  return (
    <div>
      <Card className="mb-5 flex flex-col gap-3.5 p-5">
        <Field label="Text">
          <Textarea
            value={sweep.text}
            onChange={(e) => setSweep("text", e.target.value)}
            className="min-h-[60px]"
          />
        </Field>
        <Field label="Voices">
          <div className="flex flex-wrap gap-2">
            {SPEAKERS.map((n) => (
              <VoiceChip key={n} name={n} on={!!sweep.voices[n]} onClick={() => toggleVoice(n)} />
            ))}
          </div>
        </Field>
        <div className="flex items-end gap-4">
          <div className="w-[200px]">
            <Field label="Samples per voice">
              <Slider value={sweep.n} onChange={(v) => setSweep("n", v)} min={1} max={5} step={1} />
            </Field>
          </div>
          <div className="flex-1" />
          <span className="text-xs text-txt-mute">
            {selected.length} voices × {sweep.n} = {selected.length * sweep.n} samples
          </span>
          <Button variant="primary" size="lg" icon="sparkles" onClick={genSweep}>
            Generate sweep
          </Button>
        </div>
      </Card>

      <div className="mb-3 text-xs font-bold uppercase tracking-[0.05em] text-txt-mute">
        Results ({results.length})
      </div>
      {results.length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-3.5">
          {results.map((r) => (
            <Card key={r.id} className="p-4">
              <div className="mb-3 flex items-center gap-2">
                <div className="flex h-[30px] w-[30px] items-center justify-center rounded-full bg-emerald-50">
                  <Icon name="mic" size={15} strokeWidth={2.2} className="text-emerald-600" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-bold text-txt">{r.voice}</div>
                  <div className="text-[11px] text-txt-mute">sample {r.sample}</div>
                </div>
              </div>
              <WaveformPlayer seed={r.id.length} duration={r.dur} fileName={r.file} />
            </Card>
          ))}
        </div>
      ) : (
        <Card className="rounded-[10px] border-dashed border-line-2 p-12 text-center text-sm text-txt-mute">
          Select voices and generate to compare the same line across speakers.
        </Card>
      )}
    </div>
  );
}
