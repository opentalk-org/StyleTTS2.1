import { useEffect, useState } from "react";

import { useNav } from "@/app/navStore";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Input } from "@/shared/ui/Input";
import type { IntegrationSettings, IntegrationSettingsPayload, StorageSettings, StorageSettingsPayload } from "./api";
import {
  useIntegrationSettingsActions,
  useIntegrationSettingsQuery,
  useStorageConnectionTest,
  useStorageSettingsActions,
  useStorageSettingsQuery,
} from "./query";
import { SettingsRow } from "./SettingsRow";
import { SettingsSection } from "./SettingsSection";

type SettingsForm = {
  backendUrl: string;
  storage: StorageSettingsPayload;
  integration: IntegrationSettingsPayload;
};

const DEFAULT_STORAGE_FORM: StorageSettingsPayload = {
  bucket: "runflow",
  folder: "/",
  endpoint_url: "http://127.0.0.1:9000",
  region_name: "us-east-1",
  access_key_id: "runflow",
  secret_access_key: "runflow-secret",
};

const DEFAULT_INTEGRATION_FORM: IntegrationSettingsPayload = {
  hf_token: "",
  openrouter_token: "",
  mlflow_url: "http://localhost:7860",
};

function createSettingsForm(backendUrl: string): SettingsForm {
  return { backendUrl, storage: DEFAULT_STORAGE_FORM, integration: DEFAULT_INTEGRATION_FORM };
}

function storageFormFromSettings(settings: StorageSettings): StorageSettingsPayload {
  return {
    bucket: settings.bucket,
    folder: settings.folder,
    endpoint_url: settings.endpoint_url,
    region_name: settings.region_name,
    access_key_id: settings.access_key_id,
    secret_access_key: settings.secret_access_key,
  };
}

function integrationFormFromSettings(settings: IntegrationSettings): IntegrationSettingsPayload {
  return { hf_token: settings.hf_token, openrouter_token: settings.openrouter_token, mlflow_url: settings.mlflow_url };
}

function storageFormsMatch(first: StorageSettingsPayload, second: StorageSettingsPayload): boolean {
  return first.bucket === second.bucket
    && first.folder === second.folder
    && first.endpoint_url === second.endpoint_url
    && first.region_name === second.region_name
    && first.access_key_id === second.access_key_id
    && first.secret_access_key === second.secret_access_key;
}

function integrationFormsMatch(first: IntegrationSettingsPayload, second: IntegrationSettingsPayload): boolean {
  return first.hf_token === second.hf_token
    && first.openrouter_token === second.openrouter_token
    && first.mlflow_url === second.mlflow_url;
}

function settingsFormsMatch(first: SettingsForm, second: SettingsForm): boolean {
  return first.backendUrl === second.backendUrl
    && storageFormsMatch(first.storage, second.storage)
    && integrationFormsMatch(first.integration, second.integration);
}

export function SettingsScreen() {
  const { backendUrl, setBackendUrl } = useNav();
  const storage = useStorageSettingsQuery();
  const updateStorage = useStorageSettingsActions();
  const testStorage = useStorageConnectionTest();
  const integration = useIntegrationSettingsQuery();
  const updateIntegration = useIntegrationSettingsActions();
  const initialForm = createSettingsForm(backendUrl);
  const [form, setForm] = useState<SettingsForm>(initialForm);
  const [savedForm, setSavedForm] = useState<SettingsForm>(initialForm);

  useEffect(() => {
    if (!storage.data) return;
    const nextStorageForm = storageFormFromSettings(storage.data);
    setForm((current) => {
      if (!storageFormsMatch(current.storage, savedForm.storage)) return current;
      return { ...current, storage: nextStorageForm };
    });
    setSavedForm((current) => ({ ...current, storage: nextStorageForm }));
  }, [storage.data]);

  useEffect(() => {
    if (!integration.data) return;
    const nextIntegrationForm = integrationFormFromSettings(integration.data);
    setForm((current) => {
      if (!integrationFormsMatch(current.integration, savedForm.integration)) return current;
      return { ...current, integration: nextIntegrationForm };
    });
    setSavedForm((current) => ({ ...current, integration: nextIntegrationForm }));
  }, [integration.data]);

  const storageDirty = !storageFormsMatch(form.storage, savedForm.storage);
  const integrationDirty = !integrationFormsMatch(form.integration, savedForm.integration);
  const dirty = !settingsFormsMatch(form, savedForm);
  const saving = updateStorage.isPending || updateIntegration.isPending;
  const setStorage = (key: keyof StorageSettingsPayload, value: string) => {
    setForm((current) => ({ ...current, storage: { ...current.storage, [key]: value } }));
  };
  const setIntegration = (key: keyof IntegrationSettingsPayload, value: string) => {
    setForm((current) => ({ ...current, integration: { ...current.integration, [key]: value } }));
  };
  const commitSettings = (
    submittedForm: SettingsForm,
    storageForm: StorageSettingsPayload,
    integrationForm: IntegrationSettingsPayload,
  ) => {
    const committedForm = { ...submittedForm, storage: storageForm, integration: integrationForm };
    setBackendUrl(committedForm.backendUrl);
    setForm((current) => {
      if (!settingsFormsMatch(current, submittedForm)) return current;
      return committedForm;
    });
    setSavedForm(committedForm);
    showToast("Settings saved");
  };
  const saveSettings = async () => {
    const submittedForm = form;
    let storageForm = submittedForm.storage;
    let integrationForm = submittedForm.integration;
    try {
      if (storageDirty) storageForm = storageFormFromSettings(await updateStorage.mutateAsync(submittedForm.storage));
      if (integrationDirty) integrationForm = integrationFormFromSettings(await updateIntegration.mutateAsync(submittedForm.integration));
    } catch {
      showToast("Could not save settings", undefined, "error");
      return;
    }
    commitSettings(submittedForm, storageForm, integrationForm);
  };
  const testStorageConnection = async () => {
    try {
      await testStorage.mutateAsync(form.storage);
      showToast("Storage connection works");
    } catch (error) {
      showToast("Storage connection failed", error instanceof Error ? error.message : undefined, "error");
    }
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
        <SettingsRow title="Folder" desc="Object key prefix inside the bucket. Use / to keep existing files at the bucket root.">
          <Input filled className="h-9 w-[220px] font-mono" value={form.storage.folder} onChange={(e) => setStorage("folder", e.target.value)} />
        </SettingsRow>
        <SettingsRow title="Connection" desc="Checks endpoint, bucket, region and credentials without saving changes.">
          <Button icon="refresh" disabled={testStorage.isPending} onClick={testStorageConnection}>
            {testStorage.isPending ? "Testing..." : "Test connection"}
          </Button>
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
      <SettingsSection title="Hugging Face">
        <SettingsRow title="Access token" desc="Used to download gated or private models from Hugging Face. Leave blank for public assets.">
          <Input
            filled
            type="password"
            autoComplete="off"
            placeholder="hf_…"
            className="h-9 w-[260px] font-mono"
            value={form.integration.hf_token}
            onChange={(e) => setIntegration("hf_token", e.target.value)}
          />
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="OpenRouter">
        <SettingsRow title="API key" desc="Used by the OpenRouter text-generation node to synthesize TTS prompt sentences.">
          <Input
            filled
            type="password"
            autoComplete="off"
            placeholder="sk-or-…"
            className="h-9 w-[260px] font-mono"
            value={form.integration.openrouter_token}
            onChange={(e) => setIntegration("openrouter_token", e.target.value)}
          />
        </SettingsRow>
      </SettingsSection>
      <SettingsSection title="MLflow">
        <SettingsRow title="URL" desc="Address of the MLflow experiment tracker shown on the Runs page. Leave blank to use the deployment default.">
          <Input
            filled
            placeholder="http://localhost:7860"
            className="h-9 w-[260px] font-mono"
            value={form.integration.mlflow_url}
            onChange={(e) => setIntegration("mlflow_url", e.target.value)}
          />
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
