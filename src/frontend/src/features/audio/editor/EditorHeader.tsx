import { useEffect, useRef, useState } from "react";

import { useNav } from "@/app/navStore";
import { copyText } from "@/shared/clipboard";
import { showToast } from "@/shared/feedback/Toast";
import { fmtDur } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { cn } from "@/shared/ui/cn";
import { AudioScoreInput, formatAudioScore, parseAudioScore } from "../AudioScoreInput";
import type { AudioFile } from "../api";
import { useRenameAudioFileMutation, useUpdateAudioLanguageMutation, useUpdateAudioScoreMutation, useUpdateAudioStylePromptMutation, useUpdateAudioVoicePromptMutation } from "../query";

export function EditorHeader({
  file,
  duration,
  segmentCount,
  dirty,
  onSave,
}: {
  file: AudioFile;
  duration: number;
  segmentCount: number;
  dirty: boolean;
  onSave: () => Promise<void>;
}) {
  const renameAudio = useRenameAudioFileMutation();
  const updateScore = useUpdateAudioScoreMutation();
  const updateLanguage = useUpdateAudioLanguageMutation();
  const updateStylePrompt = useUpdateAudioStylePromptMutation();
  const updateVoicePrompt = useUpdateAudioVoicePromptMutation();
  const skipNameBlur = useRef(false);
  const skipLanguageBlur = useRef(false);
  const [copied, setCopied] = useState(false);
  const [metadataCopied, setMetadataCopied] = useState(false);
  const [nameDraft, setNameDraft] = useState(file.name);
  const [scoreDraft, setScoreDraft] = useState(formatAudioScore(file.score));
  const [languageDraft, setLanguageDraft] = useState(file.language ?? "");
  const [stylePromptDraft, setStylePromptDraft] = useState(file.style_prompt ?? "");
  const [voicePromptDraft, setVoicePromptDraft] = useState(file.voice_prompt ?? "");
  const [showMetadata, setShowMetadata] = useState(false);

  useEffect(() => setNameDraft(file.name), [file.id, file.name]);
  useEffect(() => setScoreDraft(formatAudioScore(file.score)), [file.id, file.score]);
  useEffect(() => setLanguageDraft(file.language ?? ""), [file.id, file.language]);
  useEffect(() => setStylePromptDraft(file.style_prompt ?? ""), [file.id, file.style_prompt]);
  useEffect(() => setVoicePromptDraft(file.voice_prompt ?? ""), [file.id, file.voice_prompt]);

  const commitName = async () => {
    const name = nameDraft.trim();
    if (!name) {
      showToast("Audio name is required", undefined, "error");
      setNameDraft(file.name);
      return;
    }
    if (name === file.name) return;
    try {
      await renameAudio.mutateAsync({ id: file.id, name });
      showToast("Audio renamed");
    } catch {
      setNameDraft(file.name);
      showToast("Could not rename audio", undefined, "error");
    }
  };

  const commitScore = async () => {
    const raw = scoreDraft.trim();
    const score = parseAudioScore(scoreDraft);
    if (raw !== "" && score === null) {
      showToast("Score must be a number", undefined, "error");
      setScoreDraft(formatAudioScore(file.score));
      return;
    }
    if (score === file.score) return;
    if (score !== null && file.score !== null && Number(file.score.toFixed(3)) === score) return;
    try {
      await updateScore.mutateAsync({ id: file.id, score });
      showToast("Score saved");
    } catch {
      setScoreDraft(formatAudioScore(file.score));
      showToast("Could not save score", undefined, "error");
    }
  };

  const commitLanguage = async () => {
    const trimmed = languageDraft.trim();
    const language = trimmed === "" ? null : trimmed;
    if (language === (file.language ?? null)) return;
    try {
      await updateLanguage.mutateAsync({ id: file.id, language });
      showToast("Language saved");
    } catch {
      setLanguageDraft(file.language ?? "");
      showToast("Could not save language", undefined, "error");
    }
  };

  const commitStylePrompt = async () => {
    const trimmed = stylePromptDraft.trim();
    const stylePrompt = trimmed === "" ? null : trimmed;
    if (stylePrompt === (file.style_prompt ?? null)) return;
    try {
      await updateStylePrompt.mutateAsync({ id: file.id, stylePrompt });
      showToast("Style prompt saved");
    } catch {
      setStylePromptDraft(file.style_prompt ?? "");
      showToast("Could not save style prompt", undefined, "error");
    }
  };

  const commitVoicePrompt = async () => {
    const trimmed = voicePromptDraft.trim();
    const voicePrompt = trimmed === "" ? null : trimmed;
    if (voicePrompt === (file.voice_prompt ?? null)) return;
    try {
      await updateVoicePrompt.mutateAsync({ id: file.id, voicePrompt });
      showToast("Voice prompt saved");
    } catch {
      setVoicePromptDraft(file.voice_prompt ?? "");
      showToast("Could not save voice prompt", undefined, "error");
    }
  };

  const metadataJson = JSON.stringify(file.metadata, null, 2);
  return (
    <div className="relative z-30 mb-4">
      <div className="rounded-[10px] border border-line bg-panel">
        <div className="flex items-center gap-2 px-3 py-2.5">
          <Button variant="secondary" size="sm" icon="arrow-left" onClick={() => useNav.getState().go("audio")}>Back</Button>
          <input
            value={nameDraft}
            disabled={renameAudio.isPending}
            title="Rename audio"
            onChange={(event) => setNameDraft(event.target.value)}
            onBlur={() => {
              if (skipNameBlur.current) {
                skipNameBlur.current = false;
                return;
              }
              void commitName();
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur();
              if (event.key === "Escape") {
                skipNameBlur.current = true;
                setNameDraft(file.name);
                event.currentTarget.blur();
              }
            }}
            className="h-8 min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-2 font-mono text-[15px] font-bold text-txt outline-none hover:border-line-2 focus:border-blue-400 focus:bg-bg"
          />
          <span className={cn("flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs font-semibold", dirty ? "bg-amber-50 text-amber-700" : "bg-panel-2 text-txt-mute")}>
            {dirty ? <span className="h-[7px] w-[7px] rounded-full bg-amber-500" /> : <Icon name="check" size={14} strokeWidth={2.5} className="text-emerald-600" />}
            {dirty ? "Unsaved" : "Saved"}
          </span>
          <Button variant={dirty ? "primary" : "ghost"} size="sm" disabled={!dirty} onClick={() => void onSave()}>Save</Button>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-line px-3 py-2.5">
          <div className="flex min-w-0 flex-1 basis-[280px] flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] tabular-nums text-txt-mute">
            <span className="max-w-[220px] truncate rounded-full bg-panel-2 px-2 py-0.5 font-semibold text-txt-dim" title={file.speaker}>{file.speaker}</span>
            <span>{fmtDur(duration)}</span>
            <span>{file.sample_rate ? `${file.sample_rate / 1000}kHz` : "unknown rate"}</span>
            <span>{file.size_mb} MB</span>
            <span>{segmentCount} segments</span>
            <button
              title="Copy audio ID"
              onClick={async () => {
                if (!(await copyText(file.id))) return;
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
            <AudioScoreInput value={scoreDraft} disabled={updateScore.isPending} onChange={setScoreDraft} onCommit={commitScore} onCancel={() => setScoreDraft(formatAudioScore(file.score))} />
            <label className="flex h-8 items-center rounded-md border border-line-2 bg-bg pl-2.5 pr-1 focus-within:border-blue-400">
              <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">Lang</span>
              <input
                value={languageDraft}
                disabled={updateLanguage.isPending}
                placeholder="—"
                onChange={(event) => setLanguageDraft(event.target.value)}
                onBlur={() => {
                  if (skipLanguageBlur.current) {
                    skipLanguageBlur.current = false;
                    return;
                  }
                  void commitLanguage();
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                  if (event.key === "Escape") {
                    skipLanguageBlur.current = true;
                    setLanguageDraft(file.language ?? "");
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
              value={stylePromptDraft}
              disabled={updateStylePrompt.isPending}
              placeholder="Describe the speaking style…"
              onChange={(event) => setStylePromptDraft(event.target.value)}
              onBlur={() => void commitStylePrompt()}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  setStylePromptDraft(file.style_prompt ?? "");
                  event.currentTarget.blur();
                }
              }}
              className="h-6 w-full bg-transparent font-mono text-[13px] outline-none"
            />
          </label>
          <label className="flex min-w-0 flex-1 basis-[280px] flex-col gap-1 rounded-md border border-line-2 bg-bg px-2.5 py-1.5 focus-within:border-blue-400">
            <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">Voice prompt</span>
            <input
              value={voicePromptDraft}
              disabled={updateVoicePrompt.isPending}
              placeholder="Describe the voice…"
              onChange={(event) => setVoicePromptDraft(event.target.value)}
              onBlur={() => void commitVoicePrompt()}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  setVoicePromptDraft(file.voice_prompt ?? "");
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
              if (!(await copyText(metadataJson))) return;
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
