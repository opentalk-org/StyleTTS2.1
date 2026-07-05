import { useEffect, useState } from "react";

import { useNav } from "@/app/navStore";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import { Select } from "@/shared/ui/Select";
import { Toggle } from "@/shared/ui/form/Toggle";
import { useStorageSettingsActions, useStorageSettingsQuery } from "./query";
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
  const storage = useStorageSettingsQuery();
  const storageActions = useStorageSettingsActions();
  const [storageForm, setStorageForm] = useState({
    bucket: "runflow",
    endpoint_url: "http://127.0.0.1:9000",
    region_name: "us-east-1",
    access_key_id: "runflow",
    secret_access_key: "runflow-secret",
  });

  useEffect(() => {
    if (!storage.data) return;
    setStorageForm({
      bucket: storage.data.bucket,
      endpoint_url: storage.data.endpoint_url,
      region_name: storage.data.region_name,
      access_key_id: storage.data.access_key_id,
      secret_access_key: storage.data.secret_access_key,
    });
  }, [storage.data]);
  const setStorage = (key: keyof typeof storageForm, value: string) => setStorageForm((current) => ({ ...current, [key]: value }));

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
      <SettingsSection title="S3 bucket">
        <SettingsRow title="Endpoint" desc="RustFS defaults to the local S3-compatible endpoint.">
          <Input filled className="h-9 w-[260px] font-mono" value={storageForm.endpoint_url} onChange={(e) => setStorage("endpoint_url", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Bucket" desc="Bucket that stores packed audio files, checkpoints, and extra files.">
          <Input filled className="h-9 w-[220px] font-mono" value={storageForm.bucket} onChange={(e) => setStorage("bucket", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Region" desc="S3 region used by the object store client.">
          <Input filled className="h-9 w-[180px] font-mono" value={storageForm.region_name} onChange={(e) => setStorage("region_name", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Access key" desc="Credential used by backend and runners.">
          <Input filled className="h-9 w-[220px] font-mono" value={storageForm.access_key_id} onChange={(e) => setStorage("access_key_id", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Secret key" desc="Stored for local development; use environment-specific secrets outside dev.">
          <div className="flex items-center gap-2">
            <Input filled className="h-9 w-[220px] font-mono" value={storageForm.secret_access_key} onChange={(e) => setStorage("secret_access_key", e.target.value)} />
            <Button variant="primary" onClick={() => storageActions.update(storageForm)}>Save</Button>
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
