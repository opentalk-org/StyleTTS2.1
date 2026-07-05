import { useNav } from "@/app/navStore";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { Toggle } from "@/shared/ui/form/Toggle";
import { SettingsRow } from "./SettingsRow";
import { SettingsSection } from "./SettingsSection";
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
      <SettingsSection title="Backend connection">
        <SettingsRow title="Backend URL" desc="Where datasets, jobs and checkpoints stream from.">
          <Input
            filled
            className="h-9 w-[220px] font-mono"
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
          />
        </SettingsRow>
        <SettingsRow title="Poll while idle" desc="Keep fetching job status even when nothing is running.">
          <Toggle checked={s.pollWhenIdle} onChange={(v) => s.set("pollWhenIdle", v)} />
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="Defaults">
        <SettingsRow title="Auto-normalize on upload" desc="Queue a normalize job for every newly uploaded file.">
          <Toggle checked={s.autoNormalize} onChange={(v) => s.set("autoNormalize", v)} />
        </SettingsRow>
        <SettingsRow title="Default phoneme language" desc="Used to pre-fill the phonemize and synthesis forms.">
          <div className="w-[200px]">
            <Select variant="mini" value={s.defaultLang} onChange={(v) => s.set("defaultLang", v)} options={LANGS} />
          </div>
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="Safety">
        <SettingsRow title="Confirm destructive actions" desc="Ask before deleting files, datasets, voices, checkpoints, or killing jobs.">
          <Toggle checked={s.confirmDeletes} onChange={(v) => s.set("confirmDeletes", v)} />
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="Appearance">
        <SettingsRow title="Theme" desc="Interface color theme.">
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
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
