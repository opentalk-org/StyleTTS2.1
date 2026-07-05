import { useDatasets } from "../datasets/store";
import { askConfirm } from "../../shared/feedback/ConfirmDialog";
import { showToast } from "../../shared/feedback/Toast";
import { Icon } from "../../shared/icons";
import { Button } from "../../shared/ui/Button";
import { EmptyState } from "../../shared/ui/EmptyState";
import { IconButton } from "../../shared/ui/IconButton";
import { SearchInput } from "../../shared/ui/SearchInput";
import { Select } from "../../shared/ui/Select";
import { Tabs } from "../../shared/ui/Tabs";
import { VirtualTable } from "../../shared/data/VirtualTable";
import type { Voice } from "../../mock/types";
import { filterVoices, useVoices } from "./store";

export function VoicesScreen() {
  const store = useVoices();
  const datasets = useDatasets((s) => s.datasets);
  const preview = (
    <Tabs
      value={store.preview}
      onChange={(v) => store.set({ preview: v as typeof store.preview })}
      options={[
        { value: "ready", label: "Ready" },
        { value: "loading", label: "Loading" },
        { value: "error", label: "Error" },
      ]}
    />
  );

  if (store.preview === "loading") return <VoicesShell preview={preview}><VoiceSkeleton /></VoicesShell>;
  if (store.preview === "error")
    return (
      <VoicesShell preview={preview}>
        <EmptyState
          icon="alert"
          title="Couldn't reach the backend"
          description="The voices service didn't respond."
          action={
            <Button variant="primary" icon="refresh" onClick={() => store.set({ preview: "ready" })}>
              Retry
            </Button>
          }
        />
      </VoicesShell>
    );

  const list = filterVoices(store);
  const datasetOptions = [
    { value: "all", label: "All datasets" },
    ...datasets.map((d) => ({ value: d.id, label: d.name })),
  ];

  return (
    <div className="mx-auto flex h-full max-w-[960px] flex-col px-7 pb-4 pt-5">
      <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
        <Button variant="primary" icon="plus" onClick={() => { store.add(); showToast("Voice created"); }}>
          New voice
        </Button>
        <SearchInput
          value={store.query}
          onChange={(v) => store.set({ query: v })}
          placeholder={`Search ${store.voices.length.toLocaleString()} voices…`}
        />
        <Select variant="mini" value={store.dataset} onChange={(v) => store.set({ dataset: v })} options={datasetOptions} />
        <Select
          variant="mini"
          value={String(store.minSegments)}
          onChange={(v) => store.set({ minSegments: Number(v) })}
          options={[
            { value: "0", label: "Any size" },
            { value: "1", label: "Has segments" },
            { value: "50", label: "≥ 50 segments" },
            { value: "200", label: "≥ 200 segments" },
          ]}
        />
        <Select
          variant="mini"
          value={store.sort}
          onChange={(v) => store.set({ sort: v as typeof store.sort })}
          options={[
            { value: "name", label: "Sort: Name" },
            { value: "segments", label: "Sort: Most segments" },
            { value: "segments_asc", label: "Sort: Fewest segments" },
          ]}
        />
        <div className="flex-1" />
        {preview}
      </div>
      <div className="mb-2.5 text-xs tabular-nums text-txt-mute">
        {list.length.toLocaleString()} of {store.voices.length.toLocaleString()} voices
      </div>
      {list.length ? (
        <VirtualTable
          count={list.length}
          estimateRowHeight={66}
          className="flex-1"
          renderRow={(i) => <VoiceRow voice={list[i]} />}
        />
      ) : (
        <div className="rounded-[10px] border border-line bg-panel">
          <EmptyState icon="mic" title="No voices match your filters." />
        </div>
      )}
    </div>
  );
}

function VoicesShell({ preview, children }: { preview: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-[900px] px-7 pt-6">
      <div className="mb-4 flex justify-end">{preview}</div>
      <div className="rounded-[10px] border border-line bg-panel">{children}</div>
    </div>
  );
}

function VoiceSkeleton() {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3.5 p-3.5">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className="h-[92px] rounded-[10px] border border-line"
          style={{
            background: "linear-gradient(90deg,var(--color-panel) 0px,var(--color-panel-2) 200px,var(--color-panel) 400px)",
            backgroundSize: "800px 100%",
            animation: "shimmer 1.3s infinite linear",
          }}
        />
      ))}
    </div>
  );
}

function VoiceRow({ voice }: { voice: Voice }) {
  const { editId, set, rename, remove } = useVoices();
  const datasets = useDatasets((s) => s.datasets);
  const editing = editId === voice.id;

  const del = () =>
    askConfirm({
      title: "Delete voice?",
      desc: `Delete "${voice.name}". Segments keep their text but lose this voice label.`,
      danger: true,
      label: "Delete voice",
      onConfirm: () => {
        remove(voice.id);
        showToast("Voice deleted", undefined, "error");
      },
    });

  return (
    <div className="py-1">
      <div className="flex h-[58px] items-center gap-3 rounded-[9px] border border-line bg-panel px-3.5">
        <div className="flex h-[34px] w-[34px] flex-none items-center justify-center rounded-full bg-emerald-50 text-emerald-600">
          <Icon name="mic" size={16} strokeWidth={2.2} />
        </div>
        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              defaultValue={voice.name}
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") { rename(voice.id, (e.target as HTMLInputElement).value.trim() || voice.name); set({ editId: null }); }
                if (e.key === "Escape") set({ editId: null });
              }}
              onBlur={(e) => { rename(voice.id, e.target.value.trim() || voice.name); set({ editId: null }); }}
              className="h-[30px] w-full max-w-[320px] rounded-md border-2 border-blue-500 bg-panel-2 px-2.5 text-[13.5px] font-semibold text-txt outline-none"
            />
          ) : (
            <>
              <div className="truncate text-[13.5px] font-semibold text-txt">{voice.name}</div>
              <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
                {voice.datasets.map((did) => {
                  const d = datasets.find((x) => x.id === did);
                  return (
                    <span key={did} className="rounded bg-blue-50 px-1.5 py-px text-[10px] font-semibold text-blue-700">
                      {d ? d.name : did}
                    </span>
                  );
                })}
                <span className="font-mono text-[11px] text-txt-mute">{voice.id}</span>
              </div>
            </>
          )}
        </div>
        <span className="flex-none text-[12.5px] tabular-nums text-txt-dim">
          {voice.segments.toLocaleString()} seg
        </span>
        <div className="flex flex-none gap-0.5">
          <IconButton icon="edit" title="Rename" onClick={() => set({ editId: editing ? null : voice.id })} />
          <IconButton icon="trash" danger title="Delete" onClick={del} />
        </div>
      </div>
    </div>
  );
}
