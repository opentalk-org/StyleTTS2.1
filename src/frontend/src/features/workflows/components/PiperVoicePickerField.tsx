import { useEffect, useMemo, useRef, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

import { Field } from "@/shared/ui/form/Field";
import { SearchInput } from "@/shared/ui/SearchInput";
import { cn } from "@/shared/ui/cn";
import { fetchPiperVoices, type PiperVoiceCatalogItem } from "../piperApi";

type PiperVoice = PiperVoiceCatalogItem;

export function PiperVoicePickerField({
  label,
  description,
  value,
  onChange,
  catalogUrl,
}: {
  label: string;
  description?: string;
  value: unknown;
  onChange: (value: unknown) => void;
  catalogUrl: string;
}) {
  const [voices, setVoices] = useState<PiperVoice[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const selected = Array.isArray(value) ? value.map(String) : [];

  useEffect(() => {
    const controller = new AbortController();
    fetchPiperVoices(catalogUrl, controller.signal)
      .then((payload) => setVoices(Object.values(payload).sort((a, b) => a.key.localeCompare(b.key))))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => controller.abort();
  }, [catalogUrl]);

  const filtered = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return voices;
    return voices.filter((voice) =>
      [voice.key, voice.name, voice.language.code, voice.language.name_english, voice.language.country_english, voice.quality]
        .some((item) => item.toLowerCase().includes(normalized)),
    );
  }, [query, voices]);

  return (
    <Field label={`${label} (${selected.length})`} hint={description}>
      <div className="grid gap-2 rounded-md border border-line bg-panel-2 p-2">
        <SearchInput value={query} onChange={setQuery} placeholder="Search 166 Piper voices…" className="max-w-none" />
        {error ? <div className="px-2 py-3 text-[11px] text-red-600">{error}</div> : (
          <PiperVoiceList voices={filtered} selected={new Set(selected)} onToggle={(voiceId) => {
            onChange(selected.includes(voiceId) ? selected.filter((item) => item !== voiceId) : [...selected, voiceId]);
          }} />
        )}
      </div>
    </Field>
  );
}

function PiperVoiceList({ voices, selected, onToggle }: { voices: PiperVoice[]; selected: Set<string>; onToggle: (voiceId: string) => void }) {
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({ count: voices.length, getScrollElement: () => parentRef.current, estimateSize: () => 38, overscan: 8 });
  return (
    <div ref={parentRef} className="h-64 overflow-auto rounded bg-panel">
      <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((row) => {
          const voice = voices[row.index]!;
          const active = selected.has(voice.key);
          return (
            <button
              key={voice.key}
              type="button"
              onClick={() => onToggle(voice.key)}
              className={cn("absolute left-0 flex w-full items-center gap-2 px-2.5 text-left text-[12px] hover:bg-panel-2", active && "bg-blue-50")}
              style={{ height: row.size, transform: `translateY(${row.start}px)` }}
            >
              <span className={cn("h-3.5 w-3.5 shrink-0 rounded border", active ? "border-blue-500 bg-blue-500" : "border-line-2")} />
              <span className="min-w-0 flex-1 truncate font-medium">{voice.language.name_english} · {voice.name}</span>
              <span className="shrink-0 font-mono text-[10px] text-txt-mute">{voice.language.code} · {voice.quality}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
