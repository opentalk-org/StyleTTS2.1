import { useEffect, useRef } from "react";

import { backendResourceUrl } from "@/app/backend";
import { useNav } from "@/app/navStore";
import { showToast } from "@/shared/feedback/Toast";
import { useQueryClient } from "@tanstack/react-query";
import { saveAudioSegments } from "./api";
import { EditorHeader } from "./editor/EditorHeader";
import { EditorSegmentList } from "./editor/EditorSegmentList";
import { EditorTransport } from "./editor/EditorTransport";
import { useEditor } from "./editorStore";
import { AUDIO_FILES_KEY, useAudioFileQuery, useWaveformQuery, useWaveformStatusQuery } from "./query";

function segmentsSignature(segments: { id: string; start: number; end: number; text: string; phon: string; speaker: string; alignment?: { start: number }[] | null }[]): string {
  return segments
    .map((segment) => `${segment.id}:${segment.start}:${segment.end}:${segment.text}:${segment.phon}:${segment.speaker}:${segment.alignment?.length ?? 0}`)
    .sort()
    .join("|");
}

export function SegmentEditor() {
  const activeAudioFileId = useNav((state) => state.activeAudioFileId);
  const audio = useAudioFileQuery(activeAudioFileId);
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement>(null);
  const editor = useEditor();
  const {
    fileId, dur, segs, playPos, playing, speed, volume, loop, dirty, segSel,
    load, select,
  } = editor;
  const waveformStatus = useWaveformStatusQuery(activeAudioFileId);
  const waveformReady = waveformStatus.data?.status === "ready";
  const waveformPending = waveformStatus.isLoading || waveformStatus.data?.status === "pending";
  const minimapWaveform = useWaveformQuery(activeAudioFileId, 0, dur, 800, waveformReady);
  const viewWaveform = useWaveformQuery(activeAudioFileId, editor.viewStart, editor.viewEnd, 1400, waveformReady);

  useEffect(() => {
    if (!audio.data) return;
    const changedFile = audio.data.id !== fileId;
    if (!changedFile && (dirty || segmentsSignature(audio.data.segment_preview) === segmentsSignature(segs))) return;
    load(audio.data.id, audio.data.duration, audio.data.segment_preview);
  }, [audio.data, fileId, dirty, segs, load]);

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
  const saveSegments = async () => {
    const updated = await saveAudioSegments(activeAudioFileId, segs);
    load(updated.id, updated.duration, updated.segment_preview);
    await queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
    showToast("Segments saved");
  };
  const downloadAudio = () => {
    const anchor = document.createElement("a");
    anchor.href = contentUrl;
    anchor.download = file.name || "audio";
    anchor.click();
  };

  return (
    <div className="mx-auto flex h-full max-w-[1140px] flex-col px-7 pb-6 pt-[18px]">
      <audio ref={audioRef} src={contentUrl} preload="metadata" onEnded={() => {
        if (useEditor.getState().playing) useEditor.getState().togglePlay();
      }} />
      <EditorHeader file={file} duration={dur} segmentCount={segs.length} dirty={dirty} onSave={saveSegments} />
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
