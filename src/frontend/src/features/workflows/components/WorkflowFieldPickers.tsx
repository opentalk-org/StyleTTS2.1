import { useEffect, useState, type ReactNode } from "react";

import type { FieldOverride } from "@/shared/schema-form/SchemaForm";
import type { JsonSchema } from "@/shared/schema-form/types";
import { Icon } from "@/shared/icons";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Select } from "@/shared/ui/Select";
import { Field } from "@/shared/ui/form/Field";
import { cn } from "@/shared/ui/cn";
import type { AudioFile } from "../../audio/api";
import { useAudioFileQuery, useAudioFilesQuery } from "../../audio/query";
import { useDatasetsQuery } from "../../datasets/query";
import { useCheckpointsQuery } from "../../checkpoints/query";
import { PiperVoicePickerField } from "./PiperVoicePickerField";

const DATASET_FIELDS = new Set(["dataset_id", "source_dataset_id", "target_dataset_id"]);
const AUDIO_MULTI_FIELDS = new Set(["audio_file_ids"]);
const AUDIO_SINGLE_FIELDS = new Set(["audio_file_id", "reference_id"]);
const CHECKPOINT_FIELDS: Record<string, string | undefined> = {
  checkpoint_id: undefined,
  asr_checkpoint_id: "asr_bundle",
  f0_checkpoint_id: "f0_model",
  plbert_checkpoint_id: "plbert",
};

export function buildWorkflowFieldOverrides(settings: JsonSchema): Record<string, FieldOverride> {
  const overrides: Record<string, FieldOverride> = {};
  const piperCatalogUrl = settings["x-piper-catalog-url"];
  for (const name of Object.keys(settings.properties ?? {})) {
    if (name === "voice_ids" && piperCatalogUrl) {
      overrides[name] = ({ label, description, value, onChange }) => <PiperVoicePickerField label={label} description={description} value={value} onChange={onChange} catalogUrl={piperCatalogUrl} />;
    } else if (DATASET_FIELDS.has(name)) {
      overrides[name] = ({ label, description, value, onChange }) => <DatasetPickerField label={label} description={description} value={value} onChange={onChange} />;
    } else if (AUDIO_MULTI_FIELDS.has(name)) {
      overrides[name] = ({ label, description, value, onChange }) => <AudioMultiPickerField label={label} description={description} value={value} onChange={onChange} />;
    } else if (AUDIO_SINGLE_FIELDS.has(name)) {
      overrides[name] = ({ label, description, value, onChange }) => <AudioSinglePickerField label={label} description={description} value={value} onChange={onChange} />;
    } else if (name in CHECKPOINT_FIELDS) {
      const type = CHECKPOINT_FIELDS[name];
      overrides[name] = ({ label, description, value, onChange }) => <CheckpointPickerField label={label} description={description} value={value} onChange={onChange} type={type} />;
    }
  }
  return overrides;
}

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

function DatasetPickerField({ label, description, value, onChange }: { label: string; description?: string; value: unknown; onChange: (value: unknown) => void }) {
  const datasets = useDatasetsQuery();
  const list = datasets.data ?? [];
  return (
    <Field label={label} hint={description}>
      <Select
        value={String(value ?? "")}
        onChange={(dataset_id) => onChange(dataset_id)}
        options={[{ value: "", label: list.length ? "— select dataset —" : "No datasets" }, ...list.map((item) => ({ value: item.id, label: item.name }))]}
      />
    </Field>
  );
}

function CheckpointPickerField({ label, description, value, onChange, type }: { label: string; description?: string; value: unknown; onChange: (value: unknown) => void; type?: string }) {
  const checkpoints = useCheckpointsQuery();
  const list = (checkpoints.data ?? []).filter((item) => !type || item.type_ === type);
  return (
    <Field label={label} hint={description}>
      <Select
        value={String(value ?? "")}
        onChange={(checkpoint_id) => onChange(checkpoint_id)}
        options={[
          { value: "", label: list.length ? "— select checkpoint —" : "No checkpoints" },
          ...list.map((item) => ({ value: item.id, label: type ? item.name : `${item.name} · ${item.type_}` })),
        ]}
      />
    </Field>
  );
}

function Popup({ onClose, children }: { onClose: () => void; children: ReactNode }) {
  return (
    <>
      <button className="fixed inset-0 z-40 cursor-default" aria-label="Close" onClick={onClose} />
      <div className="absolute left-0 right-0 top-[calc(100%+4px)] z-50 rounded-md border border-line bg-panel p-2 shadow-[0_18px_60px_rgba(17,24,39,0.22)]">
        {children}
      </div>
    </>
  );
}

function AudioSearchList({ onPick, exclude }: { onPick: (audio: AudioFile) => void; exclude?: Set<string> }) {
  const [query, setQuery] = useState("");
  const debounced = useDebounced(query, 250);
  const audio = useAudioFilesQuery({ query: debounced, language: "", dataset: "all", sort: "updated", limit: 40, offset: 0 });
  const rows = audio.data?.rows ?? [];
  const total = audio.data?.total ?? 0;

  return (
    <div className="grid gap-2">
      <SearchInput value={query} onChange={setQuery} placeholder="Search audio by name…" className="max-w-none" />
      <div className="max-h-64 overflow-auto rounded-md">
        {rows.length ? (
          rows.map((item) => {
            const disabled = exclude?.has(item.id) ?? false;
            return (
              <button
                key={item.id}
                disabled={disabled}
                onClick={() => onPick(item)}
                className={cn(
                  "flex w-full items-center gap-2 rounded px-2.5 py-1.5 text-left text-[12.5px]",
                  disabled ? "cursor-default opacity-40" : "hover:bg-panel-2",
                )}
              >
                <span className="min-w-0 flex-1 truncate font-medium text-txt">{item.name}</span>
                <span className="shrink-0 text-[11px] tabular-nums text-txt-mute">
                  {item.annotations.speaker_id ? `${item.annotations.speaker_id} · ` : ""}
                  {item.duration.toFixed(1)}s
                </span>
              </button>
            );
          })
        ) : (
          <div className="px-2.5 py-3 text-[12px] text-txt-mute">{audio.isLoading ? "Searching…" : "No matching audio."}</div>
        )}
      </div>
      {total > rows.length ? (
        <div className="px-1 text-[11px] text-txt-mute">
          Showing {rows.length} of {total.toLocaleString()} — refine your search.
        </div>
      ) : null}
    </div>
  );
}

function AudioChip({ id, onRemove }: { id: string; onRemove: () => void }) {
  const audio = useAudioFileQuery(id);
  return (
    <span className="inline-flex max-w-full items-center gap-1 rounded-full bg-panel-2 py-0.5 pl-2.5 pr-1 text-[12px]">
      <span className="max-w-[220px] truncate text-txt-dim">{audio.data?.name ?? `${id.slice(0, 8)}…`}</span>
      <button onClick={onRemove} title="Remove" className="flex h-4 w-4 items-center justify-center rounded-full text-txt-mute hover:bg-panel-3">
        <Icon name="x" size={11} strokeWidth={2.4} />
      </button>
    </span>
  );
}

function TriggerButton({ open, onClick, children }: { open: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex h-9 w-full items-center justify-between gap-2 rounded-md border-2 bg-panel-2 px-3 text-[13px] text-txt",
        open ? "border-blue-500" : "border-transparent",
      )}
    >
      <span className="min-w-0 truncate">{children}</span>
      <Icon name="chevron-down" size={14} strokeWidth={2.2} className="shrink-0 text-txt-mute" />
    </button>
  );
}

export function AudioSinglePickerField({ label, description, value, onChange }: { label: string; description?: string; value: unknown; onChange: (value: unknown) => void }) {
  const [open, setOpen] = useState(false);
  const current = String(value ?? "");
  const audio = useAudioFileQuery(current || null);
  return (
    <Field label={label} hint={description}>
      <div className="relative">
        <TriggerButton open={open} onClick={() => setOpen((v) => !v)}>
          {current ? (audio.data?.name ?? current) : "Choose audio…"}
        </TriggerButton>
        {open ? (
          <Popup onClose={() => setOpen(false)}>
            <AudioSearchList
              onPick={(item) => {
                onChange(item.id);
                setOpen(false);
              }}
            />
          </Popup>
        ) : null}
      </div>
    </Field>
  );
}

function AudioMultiPickerField({ label, description, value, onChange }: { label: string; description?: string; value: unknown; onChange: (value: unknown) => void }) {
  const [open, setOpen] = useState(false);
  const ids = Array.isArray(value) ? value.map(String) : [];
  const add = (item: AudioFile) => {
    if (!ids.includes(item.id)) onChange([...ids, item.id]);
  };
  const remove = (id: string) => onChange(ids.filter((item) => item !== id));

  return (
    <Field label={`${label} (${ids.length})`} hint={description}>
      {ids.length ? (
        <div className="mb-1.5 flex flex-wrap gap-1.5">
          {ids.map((id) => (
            <AudioChip key={id} id={id} onRemove={() => remove(id)} />
          ))}
        </div>
      ) : null}
      <div className="relative">
        <TriggerButton open={open} onClick={() => setOpen((v) => !v)}>
          Add audio…
        </TriggerButton>
        {open ? (
          <Popup onClose={() => setOpen(false)}>
            <AudioSearchList onPick={add} exclude={new Set(ids)} />
          </Popup>
        ) : null}
      </div>
    </Field>
  );
}
