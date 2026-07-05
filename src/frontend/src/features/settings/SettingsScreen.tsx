import { useEffect, useState } from "react";

import { useNav } from "@/app/navStore";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import type { StorageSettings, StorageSettingsPayload } from "./api";
import { useStorageSettingsActions, useStorageSettingsQuery } from "./query";
import { SettingsRow } from "./SettingsRow";
import { SettingsSection } from "./SettingsSection";

type SettingsForm = {
  backendUrl: string;
  storage: StorageSettingsPayload;
};

const DEFAULT_STORAGE_FORM: StorageSettingsPayload = {
  bucket: "runflow",
  endpoint_url: "http://127.0.0.1:9000",
  region_name: "us-east-1",
  access_key_id: "runflow",
  secret_access_key: "runflow-secret",
};

function createSettingsForm(backendUrl: string, storage: StorageSettingsPayload): SettingsForm {
  return { backendUrl, storage };
}

function storageFormFromSettings(settings: StorageSettings): StorageSettingsPayload {
  return {
    bucket: settings.bucket,
    endpoint_url: settings.endpoint_url,
    region_name: settings.region_name,
    access_key_id: settings.access_key_id,
    secret_access_key: settings.secret_access_key,
  };
}

function storageFormsMatch(first: StorageSettingsPayload, second: StorageSettingsPayload): boolean {
  return first.bucket === second.bucket
    && first.endpoint_url === second.endpoint_url
    && first.region_name === second.region_name
    && first.access_key_id === second.access_key_id
    && first.secret_access_key === second.secret_access_key;
}

function settingsFormsMatch(first: SettingsForm, second: SettingsForm): boolean {
  return first.backendUrl === second.backendUrl
    && storageFormsMatch(first.storage, second.storage);
}

export function SettingsScreen() {
  const { backendUrl, setBackendUrl } = useNav();
  const storage = useStorageSettingsQuery();
  const updateStorage = useStorageSettingsActions();
  const initialForm = createSettingsForm(backendUrl, DEFAULT_STORAGE_FORM);
  const [form, setForm] = useState<SettingsForm>(initialForm);
  const [savedForm, setSavedForm] = useState<SettingsForm>(initialForm);

  useEffect(() => {
    if (!storage.data) return;
    const nextStorageForm = storageFormFromSettings(storage.data);
    const previousSavedStorage = savedForm.storage;
    setForm((current) => {
      if (!storageFormsMatch(current.storage, previousSavedStorage)) return current;
      return { ...current, storage: nextStorageForm };
    });
    setSavedForm((current) => ({ ...current, storage: nextStorageForm }));
  }, [storage.data]);

  const storageDirty = !storageFormsMatch(form.storage, savedForm.storage);
  const dirty = !settingsFormsMatch(form, savedForm);
  const saving = updateStorage.isPending;
  const setStorage = (key: keyof StorageSettingsPayload, value: string) => {
    setForm((current) => ({ ...current, storage: { ...current.storage, [key]: value } }));
  };
  const commitSettings = (submittedForm: SettingsForm, storageForm: StorageSettingsPayload) => {
    const committedForm = { ...submittedForm, storage: storageForm };
    setBackendUrl(committedForm.backendUrl);
    setForm((current) => {
      if (!settingsFormsMatch(current, submittedForm)) return current;
      return committedForm;
    });
    setSavedForm(committedForm);
    showToast("Settings saved");
  };
  const saveSettings = () => {
    const submittedForm = form;
    if (storageDirty) {
      updateStorage.mutate(submittedForm.storage, {
        onSuccess: (updatedStorage) => commitSettings(submittedForm, storageFormFromSettings(updatedStorage)),
      });
      return;
    }
    commitSettings(submittedForm, submittedForm.storage);
  };

  return (
    <div className="mx-auto max-w-[720px] px-7 pb-16 pt-6">
      <SettingsSection title="Backend connection">
        <SettingsRow title="Backend URL" desc="Where datasets, jobs and checkpoints stream from.">
          <Input
            filled
            className="h-9 w-[220px] font-mono"
            value={form.backendUrl}
            onChange={(e) => setForm((current) => ({ ...current, backendUrl: e.target.value }))}
          />
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="S3 bucket">
        <SettingsRow title="Endpoint" desc="RustFS defaults to the local S3-compatible endpoint.">
          <Input filled className="h-9 w-[260px] font-mono" value={form.storage.endpoint_url} onChange={(e) => setStorage("endpoint_url", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Bucket" desc="Bucket that stores packed audio files, checkpoints, and extra files.">
          <Input filled className="h-9 w-[220px] font-mono" value={form.storage.bucket} onChange={(e) => setStorage("bucket", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Region" desc="S3 region used by the object store client.">
          <Input filled className="h-9 w-[180px] font-mono" value={form.storage.region_name} onChange={(e) => setStorage("region_name", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Access key" desc="Credential used by backend and runners.">
          <Input filled className="h-9 w-[220px] font-mono" value={form.storage.access_key_id} onChange={(e) => setStorage("access_key_id", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Secret key" desc="Stored for local development; use environment-specific secrets outside dev.">
          <Input filled className="h-9 w-[220px] font-mono" value={form.storage.secret_access_key} onChange={(e) => setStorage("secret_access_key", e.target.value)} />
        </SettingsRow>
      </SettingsSection>
      <div className="sticky bottom-0 -mx-7 mt-3 flex items-center justify-between border-t border-line bg-app/95 px-7 py-4 backdrop-blur">
        <div className="text-xs font-medium text-txt-mute">
          {dirty ? "Unsaved changes" : "All changes saved"}
        </div>
        <Button variant="primary" disabled={!dirty || saving} onClick={saveSettings}>
          {saving ? "Saving..." : "Save settings"}
        </Button>
      </div>
    </div>
  );
}
