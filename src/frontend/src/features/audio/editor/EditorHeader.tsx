import { useState } from "react";

import { copyText } from "@/shared/clipboard";
import { showToast } from "@/shared/feedback/Toast";
import { fmtDur } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { AudioScoreInput, formatAudioScore } from "../AudioScoreInput";
import type { AudioFile } from "../api";
import { useNavigate } from "react-router-dom";

export type EditorHeaderDraft = {
  name: string;
  score: string;
  language: string;
  stylePrompt: string;
  voicePrompt: string;
};

export function EditorHeader({
  file,
  duration,
  segmentCount,
  draft,
  dirty,
  saving,
  onDraftChange,
  onSave,
}: {
  file: AudioFile;
  duration: number;
  segmentCount: number;
  draft: EditorHeaderDraft;
  dirty: boolean;
  saving: boolean;
  onDraftChange: (draft: EditorHeaderDraft) => void;
  onSave: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);
  const [metadataCopied, setMetadataCopied] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);

  const metadataJson = JSON.stringify(file.annotations.metadata, null, 2);
  return (
    <div className="relative z-30 mb-4">
      <div className="rounded-[10px] border border-line bg-panel">
        <div className="flex items-center gap-2 px-3 py-2.5">
          <Button variant="secondary" size="sm" icon="arrow-left" onClick={() => void navigate("/audio")}>Back</Button>
          <input
            value={draft.name}
            disabled={saving}
            title="Rename audio"
            onChange={(event) => onDraftChange({ ...draft, name: event.target.value })}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") {
                onDraftChange({ ...draft, name: file.name });
                event.currentTarget.blur();
              }
            }}
            className="h-8 min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-2 font-mono text-[15px] font-bold text-txt outline-none hover:border-line-2 focus:border-blue-400 focus:bg-bg"
          />
          <span className={cn("flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs font-semibold", dirty ? "bg-amber-50 text-amber-700" : "bg-panel-2 text-txt-mute")}>
            {dirty ? <span className="h-[7px] w-[7px] rounded-full bg-amber-500" /> : <Icon name="check" size={14} strokeWidth={2.5} className="text-emerald-600" />}
            {dirty ? "Unsaved" : "Saved"}
          </span>
          <Button variant={dirty ? "primary" : "ghost"} size="sm" disabled={!dirty || saving} onClick={() => void onSave()}>{saving ? "Saving…" : "Save"}</Button>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-line px-3 py-2.5">
          <div className="flex min-w-0 flex-1 basis-[280px] flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] tabular-nums text-txt-mute">
            <span>{fmtDur(duration)}</span>
            <span>{file.sample_rate ? `${file.sample_rate / 1000}kHz` : "unknown rate"}</span>
            <span>{file.size_mb} MB</span>
            <span>{segmentCount} segments</span>
            <button
              title="Copy audio ID"
              onClick={async () => {
                await copyText(file.id);
                setCopied(true);
                showToast("Audio ID copied");
                window.setTimeout(() => setCopied(false), 1200);
              }}
              className={cn("inline-flex max-w-[200px] items-center gap-1 font-mono", copied ? "text-emerald-600" : "text-txt-dim hover:text-txt")}
            >
              <Icon name={copied ? "check" : "copy"} size={11} strokeWidth={2.4} />
              <span className="truncate">{copied ? "Copied" : file.id}</span>
            </button>
          </div>
          <div className="flex items-center gap-2">
            <AudioScoreInput value={draft.score} disabled={saving} onChange={(score) => onDraftChange({ ...draft, score })} onCancel={() => onDraftChange({ ...draft, score: formatAudioScore(file.annotations.score) })} />
            <label className="flex h-8 items-center rounded-md border border-line-2 bg-bg pl-2.5 pr-1 focus-within:border-blue-400">
              <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">Lang</span>
              <input
                value={draft.language}
                disabled={saving}
                placeholder="—"
                onChange={(event) => onDraftChange({ ...draft, language: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                  if (event.key === "Escape") {
                    onDraftChange({ ...draft, language: file.language ?? "" });
                    event.currentTarget.blur();
                  }
                }}
                className="h-6 w-20 bg-transparent px-1 font-mono text-[13px] font-semibold outline-none"
              />
            </label>
            <Button variant={showMetadata ? "secondary" : "ghost"} size="sm" icon="file-audio" onClick={() => setShowMetadata((value) => !value)}>Metadata</Button>
          </div>
        </div>
        <div className="flex flex-wrap items-stretch gap-2 border-t border-line px-3 py-2.5">
          <label className="flex min-w-0 flex-1 basis-[280px] flex-col gap-1 rounded-md border border-line-2 bg-bg px-2.5 py-1.5 focus-within:border-blue-400">
            <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">Style prompt</span>
            <input
              value={draft.stylePrompt}
              disabled={saving}
              placeholder="Describe the speaking style…"
              onChange={(event) => onDraftChange({ ...draft, stylePrompt: event.target.value })}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  onDraftChange({ ...draft, stylePrompt: file.style_prompt ?? "" });
                  event.currentTarget.blur();
                }
              }}
              className="h-6 w-full bg-transparent font-mono text-[13px] outline-none"
            />
          </label>
          <label className="flex min-w-0 flex-1 basis-[280px] flex-col gap-1 rounded-md border border-line-2 bg-bg px-2.5 py-1.5 focus-within:border-blue-400">
            <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">Voice prompt</span>
            <input
              value={draft.voicePrompt}
              disabled={saving}
              placeholder="Describe the voice…"
              onChange={(event) => onDraftChange({ ...draft, voicePrompt: event.target.value })}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  onDraftChange({ ...draft, voicePrompt: file.voice_prompt ?? "" });
                  event.currentTarget.blur();
                }
              }}
              className="h-6 w-full bg-transparent font-mono text-[13px] outline-none"
            />
          </label>
        </div>
      </div>
      {showMetadata ? (
        <div className="absolute right-0 top-full z-30 mt-2 max-h-[70vh] w-full max-w-[560px] overflow-hidden rounded-[10px] border border-line bg-panel shadow-xl">
          <div className="flex items-center justify-between border-b border-line px-4 py-2">
            <div className="text-[13px] font-bold text-txt">Audio metadata</div>
            <Button variant="ghost" size="sm" icon={metadataCopied ? "check" : "copy"} onClick={async () => {
              await copyText(metadataJson);
              setMetadataCopied(true);
              showToast("Metadata copied");
              window.setTimeout(() => setMetadataCopied(false), 1200);
            }}>{metadataCopied ? "Copied" : "Copy"}</Button>
          </div>
          <pre className="max-h-[calc(70vh-44px)] overflow-auto bg-bg p-4 font-mono text-[11px] leading-relaxed text-txt-dim">{metadataJson}</pre>
        </div>
      ) : null}
    </div>
  );
}
