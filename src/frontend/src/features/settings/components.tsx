import type { ReactNode } from "react";

import { useNav } from "../../app/navStore";
import { Input } from "../../shared/ui/Input";
import { Select } from "../../shared/ui/Select";
import { Toggle } from "../../shared/ui/controls";
import { useSettings } from "./store";

const LANGS = [
  { value: "en-us", label: "English (US)" },
  { value: "en-gb", label: "English (UK)" },
  { value: "es", label: "Spanish" },
  { value: "de", label: "German" },
  { value: "fr", label: "French" },
  { value: "ja", label: "Japanese" },
];

export function SettingsScreen() {
  const s = useSettings();
  const { backendUrl, setBackendUrl } = useNav();

  return (
    <div className="mx-auto max-w-[720px] px-7 pb-16 pt-6">
      <Section title="Backend connection">
        <Row title="Backend URL" desc="Where datasets, jobs and checkpoints stream from.">
          <Input
            filled
            className="h-9 w-[220px] font-mono"
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
          />
        </Row>
        <Row title="Poll while idle" desc="Keep fetching job status even when nothing is running.">
          <Toggle checked={s.pollWhenIdle} onChange={(v) => s.set("pollWhenIdle", v)} />
        </Row>
      </Section>
      <Section title="Defaults">
        <Row title="Auto-normalize on upload" desc="Queue a normalize job for every newly uploaded file.">
          <Toggle checked={s.autoNormalize} onChange={(v) => s.set("autoNormalize", v)} />
        </Row>
        <Row title="Default phoneme language" desc="Used to pre-fill the phonemize and synthesis forms.">
          <div className="w-[200px]">
            <Select variant="mini" value={s.defaultLang} onChange={(v) => s.set("defaultLang", v)} options={LANGS} />
          </div>
        </Row>
      </Section>
      <Section title="Safety">
        <Row title="Confirm destructive actions" desc="Ask before deleting files, datasets, voices, checkpoints, or killing jobs.">
          <Toggle checked={s.confirmDeletes} onChange={(v) => s.set("confirmDeletes", v)} />
        </Row>
      </Section>
      <Section title="Appearance">
        <Row title="Theme" desc="Interface color theme.">
          <div className="w-[200px]">
            <Select
              variant="mini"
              value={s.theme}
              onChange={(v) => s.set("theme", v as typeof s.theme)}
              options={[
                { value: "light", label: "Light" },
                { value: "system", label: "System (soon)" },
                { value: "dark", label: "Dark (soon)" },
              ]}
            />
          </div>
        </Row>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mb-4 rounded-[10px] border border-line bg-panel px-5 pb-2 pt-1.5">
      <div className="pt-3.5 pb-0.5 text-[11px] font-bold uppercase tracking-wider text-blue-500">
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ title, desc, children }: { title: string; desc: string; children: ReactNode }) {
  return (
    <div className="flex items-center gap-4 border-b border-line py-4 last:border-b-0">
      <div className="flex-1">
        <div className="text-[13.5px] font-semibold text-txt">{title}</div>
        <div className="mt-0.5 text-xs text-txt-mute">{desc}</div>
      </div>
      {children}
    </div>
  );
}
