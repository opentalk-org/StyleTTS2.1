import { useEffect, useRef, useState } from "react";

import { backendResourceUrl } from "@/app/backend";
import { useNav } from "@/app/navStore";
import { showToast } from "@/shared/feedback/Toast";
import { useQueryClient } from "@tanstack/react-query";
import { renameAudioFile, saveAudioSegments, updateAudioLanguage, updateAudioScore, updateAudioStylePrompt, updateAudioVoicePrompt, type AudioFile } from "./api";
import { EditorHeader, type EditorHeaderDraft } from "./editor/EditorHeader";
import { EditorSegmentList } from "./editor/EditorSegmentList";
import { EditorTransport } from "./editor/EditorTransport";
import { useEditor } from "./editorStore";
import { parseAudioScore } from "./AudioScoreInput";
import { AUDIO_FILES_KEY, useAudioFileQuery, useWaveformQuery, useWaveformStatusQuery } from "./query";

function segmentsSignature(segments: { id: string; start: number; end: number; text: string; phon: string; speaker: string; alignment?: { start: number }[] | null }[]): string {
  return segments
    .map((segment) => `${segment.id}:${segment.start}:${segment.end}:${segment.text}:${segment.phon}:${segment.speaker}:${segment.alignment?.length ?? 0}`)
    .sort()
    .join("|");
}

function headerDraft(file: AudioFile): EditorHeaderDraft {
  return {
    name: file.name,
    score: file.score === null ? "" : file.score.toFixed(3),
    language: file.language ?? "",
    stylePrompt: file.style_prompt ?? "",
    voicePrompt: file.voice_prompt ?? "",
  };
}

function optionalText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

function scoreMatchesFile(score: number | null, file: AudioFile): boolean {
  return score === file.score || (score !== null && file.score !== null && Number(file.score.toFixed(3)) === score);
}

function draftMatchesFile(draft: EditorHeaderDraft, file: AudioFile): boolean {
  const score = parseAudioScore(draft.score);
  return draft.name.trim() === file.name
    && scoreMatchesFile(score, file)
    && optionalText(draft.language) === file.language
    && optionalText(draft.stylePrompt) === file.style_prompt
    && optionalText(draft.voicePrompt) === file.voice_prompt;
}

export function SegmentEditor() {
  const activeAudioFileId = useNav((state) => state.activeAudioFileId);
  const audio = useAudioFileQuery(activeAudioFileId);
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement>(null);
  const draftFileId = useRef<string | null>(null);
  const [draft, setDraft] = useState<EditorHeaderDraft>({ name: "", score: "", language: "", stylePrompt: "", voicePrompt: "" });
  const [saving, setSaving] = useState(false);
  const editor = useEditor();
  const {
    fileId, dur, segs, playPos, playing, speed, volume, loop, dirty: segmentsDirty, segSel,
    load, select,
  } = editor;
  const waveformStatus = useWaveformStatusQuery(activeAudioFileId);
  const waveformReady = waveformStatus.data?.status === "ready";
  const waveformPending = waveformStatus.isLoading || waveformStatus.data?.status === "pending";
  const minimapWaveform = useWaveformQuery(activeAudioFileId, 0, dur, 800, waveformReady);
  const viewWaveform = useWaveformQuery(activeAudioFileId, editor.viewStart, editor.viewEnd, 1400, waveformReady);
  const metadataDirty = audio.data ? !draftMatchesFile(draft, audio.data) : false;
  const dirty = segmentsDirty || metadataDirty;

  useEffect(() => {
    if (!audio.data) return;
    if (draftFileId.current === audio.data.id && metadataDirty) return;
    draftFileId.current = audio.data.id;
    setDraft(headerDraft(audio.data));
  }, [audio.data, metadataDirty]);

  useEffect(() => {
    if (!audio.data) return;
    const changedFile = audio.data.id !== fileId;
    if (!changedFile && (segmentsDirty || segmentsSignature(audio.data.segment_preview) === segmentsSignature(segs))) return;
    load(audio.data.id, audio.data.duration, audio.data.segment_preview);
  }, [audio.data, fileId, segmentsDirty, segs, load]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    element.volume = volume;
    element.playbackRate = speed;
  }, [speed, volume]);

  useEffect(() => {
    const element = audioRef.current;
    if (element && Math.abs(element.currentTime - playPos) > 0.25) element.currentTime = playPos;
  }, [playPos]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    if (playing) void element.play().catch(() => useEditor.getState().togglePlay());
    else element.pause();
  }, [playing]);

  useEffect(() => {
    const current = segs.find((segment) => segment.id === segSel);
    if (current && playPos >= current.start && playPos <= current.end) return;
    const at = segs.find((segment) => playPos >= segment.start && playPos < segment.end)
      ?? segs.find((segment) => playPos >= segment.start && playPos <= segment.end);
    if (at && at.id !== segSel) select(at.id);
  }, [playPos, segs, segSel, select]);

  useEffect(() => {
    const element = audioRef.current;
    if (!playing || !element) return;
    let frame = 0;
    const tick = () => {
      const state = useEditor.getState();
      let next = element.currentTime;
      if (state.loop) {
        const selected = state.segs.find((segment) => segment.id === state.segSel);
        const start = selected ? selected.start : 0;
        const end = selected ? selected.end : state.dur;
        if (next >= end) {
          element.currentTime = start;
          next = start;
        }
      }
      state.seek(next);
      state.followPlayhead();
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, loop]);

  if (activeAudioFileId === null) return <></>;
  if (audio.isLoading) return <div className="p-7 text-sm text-txt-mute">Loading segment editor...</div>;
  if (audio.isError || !audio.data || fileId !== activeAudioFileId) return <div className="p-7 text-sm text-txt-mute">Audio file is unavailable.</div>;

  const file = audio.data;
  const contentUrl = backendResourceUrl(`/audio-files/${encodeURIComponent(activeAudioFileId)}/content`);
  const saveChanges = async () => {
    const name = draft.name.trim();
    const score = parseAudioScore(draft.score);
    if (name === "") {
      showToast("Audio name is required", undefined, "error");
      return;
    }
    if (draft.score.trim() !== "" && score === null) {
      showToast("Score must be a number", undefined, "error");
      return;
    }

    setSaving(true);
    try {
      const metadataSaves: Promise<AudioFile>[] = [];
      if (name !== file.name) metadataSaves.push(renameAudioFile(file.id, name));
      if (!scoreMatchesFile(score, file)) metadataSaves.push(updateAudioScore(file.id, score));
      if (optionalText(draft.language) !== file.language) metadataSaves.push(updateAudioLanguage(file.id, optionalText(draft.language)));
      if (optionalText(draft.stylePrompt) !== file.style_prompt) metadataSaves.push(updateAudioStylePrompt(file.id, optionalText(draft.stylePrompt)));
      if (optionalText(draft.voicePrompt) !== file.voice_prompt) metadataSaves.push(updateAudioVoicePrompt(file.id, optionalText(draft.voicePrompt)));
      await Promise.all(metadataSaves);

      if (segmentsDirty) {
        const updated = await saveAudioSegments(activeAudioFileId, segs);
        load(updated.id, updated.duration, updated.segment_preview);
      }
      setDraft({ name, score: score === null ? "" : score.toFixed(3), language: optionalText(draft.language) ?? "", stylePrompt: optionalText(draft.stylePrompt) ?? "", voicePrompt: optionalText(draft.voicePrompt) ?? "" });
      await queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
      showToast("Changes saved");
    } catch {
      await queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
      showToast("Could not save changes", undefined, "error");
    } finally {
      setSaving(false);
    }
  };
  const downloadAudio = () => {
    const anchor = document.createElement("a");
    anchor.href = contentUrl;
    anchor.download = file.name || "audio";
    anchor.click();
  };

  return (
    <div className="mx-auto flex min-h-full max-w-[1140px] flex-col px-7 pb-6 pt-[18px]">
      <audio ref={audioRef} src={contentUrl} preload="metadata" onEnded={() => {
        if (useEditor.getState().playing) useEditor.getState().togglePlay();
      }} />
      <EditorHeader file={file} duration={dur} segmentCount={segs.length} draft={draft} dirty={dirty} saving={saving} onDraftChange={setDraft} onSave={saveChanges} />
      <EditorTransport
        waveformPending={waveformPending}
        minimapPeaks={minimapWaveform.data?.peaks}
        viewPeaks={viewWaveform.data?.peaks}
        seed={hashSeed(activeAudioFileId)}
        onDownload={downloadAudio}
      />
      <EditorSegmentList />
    </div>
  );
}

function hashSeed(value: string): number {
  let out = 0;
  for (const char of value) out = (out * 31 + char.charCodeAt(0)) >>> 0;
  return out || 1;
}
