import { useEffect, useRef, useState } from "react";

import { backendResourceUrl } from "@/app/backend";
import { useNav } from "@/app/navStore";
import { copyText } from "@/shared/clipboard";
import { showToast } from "@/shared/feedback/Toast";
import { fmtClock, fmtDur } from "@/shared/format";
import { Icon, type IconName } from "@/shared/icons";
import { VirtualTable } from "@/shared/data/VirtualTable";
import { Button } from "@/shared/ui/Button";
import { SearchInput } from "@/shared/ui/SearchInput";
import { Slider } from "@/shared/ui/form/Slider";
import { cn } from "@/shared/ui/cn";
import { useQueryClient } from "@tanstack/react-query";
import { saveAudioSegments } from "./api";
import { SegmentRow } from "./SegmentRow";
import { SegmentTimeline } from "./SegmentTimeline";
import { useEditor } from "./editorStore";
import { AUDIO_FILES_KEY, useAudioFileQuery, useWaveformQuery } from "./query";

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];

function TBtn({ icon, title, big, flip, onClick }: { icon: IconName; title: string; big?: boolean; flip?: boolean; onClick: () => void }) {
  return (
    <button
      title={title}
      onClick={onClick}
      className={cn(
        "flex items-center justify-center rounded-md bg-panel-2 text-txt hover:bg-panel-3",
        big ? "h-9 w-9" : "h-9 w-9",
        flip && "-scale-x-100",
      )}
    >
      <Icon name={icon} size={16} strokeWidth={2.2} />
    </button>
  );
}

export function SegmentEditor() {
  const activeAudioFileId = useNav((s) => s.activeAudioFileId);
  const audio = useAudioFileQuery(activeAudioFileId);
  const queryClient = useQueryClient();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [copied, setCopied] = useState(false);
  const ed = useEditor();
  const {
    fileId, dur, segs, playPos, playing, speed, volume, loop, abA, abB, viewStart, viewEnd, dirty, segSel, segQuery,
    load, seek, togglePlay, setSpeed, setVolume, toggleLoop, setA, setB, clearAB, setView, zoomIn, zoomOut, select, setSegTime, setQuery, addSeg,
  } = ed;
  const minimapWaveform = useWaveformQuery(activeAudioFileId, 0, dur, 800);
  const viewWaveform = useWaveformQuery(activeAudioFileId, viewStart, viewEnd, 1400);

  useEffect(() => {
    if (!audio.data || audio.data.id === fileId) return;
    load(audio.data.id, audio.data.duration, audio.data.segment_preview);
  }, [audio.data, fileId, load]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    element.volume = volume;
    element.playbackRate = speed;
  }, [speed, volume]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    if (Math.abs(element.currentTime - playPos) > 0.25) element.currentTime = playPos;
  }, [playPos]);

  useEffect(() => {
    const element = audioRef.current;
    if (!element) return;
    if (playing) void element.play().catch(() => useEditor.getState().togglePlay());
    else element.pause();
  }, [playing]);

  // Smooth playhead: while playing, sample the media clock every animation frame
  // (~60fps) rather than leaning on the audio element's `timeupdate` event, which
  // only fires every ~250-350ms and makes the cursor and time readout visibly jump.
  useEffect(() => {
    const element = audioRef.current;
    if (!playing || !element) return;
    let raf = 0;
    const tick = () => {
      const state = useEditor.getState();
      let next = element.currentTime;
      if (state.loop && state.abA != null && state.abB != null) {
        const lo = Math.min(state.abA, state.abB);
        const hi = Math.max(state.abA, state.abB);
        if (next >= hi) {
          element.currentTime = lo;
          next = lo;
        }
      }
      state.seek(next);
      state.followPlayhead();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  if (activeAudioFileId === null) return <></>;
  if (audio.isLoading) return <div className="p-7 text-sm text-txt-mute">Loading segment editor...</div>;
  if (audio.isError || !audio.data || fileId !== activeAudioFileId) return <div className="p-7 text-sm text-txt-mute">Audio file is unavailable.</div>;

  const file = audio.data;
  const q = segQuery.trim().toLowerCase();
  const vis = q ? segs.filter((g) => g.text.toLowerCase().includes(q) || g.phon.toLowerCase().includes(q)) : segs;
  const contentUrl = backendResourceUrl(`/audio-files/${encodeURIComponent(activeAudioFileId)}/content`);
  const seed = hashSeed(activeAudioFileId);

  const saveSegments = async () => {
    const updated = await saveAudioSegments(activeAudioFileId, segs);
    load(updated.id, updated.duration, updated.segment_preview);
    await queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
    showToast("Segments saved");
  };

  const abBtn = (label: string, on: boolean, onClick: () => void) => (
    <button
      onClick={onClick}
      title={`Set loop point ${label}`}
      className={cn("h-7 rounded px-2.5 text-xs font-bold", on ? "bg-emerald-50 text-emerald-700" : "text-txt-dim")}
    >
      {label}
    </button>
  );

  return (
    <div className="mx-auto flex h-full max-w-[1140px] flex-col px-7 pb-6 pt-[18px]">
      <audio
        ref={audioRef}
        src={contentUrl}
        preload="metadata"
        onEnded={() => {
          if (useEditor.getState().playing) useEditor.getState().togglePlay();
        }}
      />
      {/* header */}
      <div className="mb-4 flex items-center gap-3.5">
        <Button variant="secondary" icon="arrow-left" onClick={() => useNav.getState().go("audio")}>
          Audio Files
        </Button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2.5">
            <span className="truncate font-mono text-[17px] font-bold tracking-tight">{file.name}</span>
            <span className="rounded-full bg-panel-2 px-2 py-0.5 text-[11px] font-semibold text-txt-dim">{file.speaker}</span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-1.5 text-xs tabular-nums text-txt-mute">
            <span>{fmtDur(dur)} · {file.sample_rate ? `${file.sample_rate / 1000}kHz` : "unknown rate"} · {file.size_mb} MB · {segs.length} segments</span>
            <span className="text-line">·</span>
            <button
              title="Copy audio ID"
              onClick={async () => {
                if (!(await copyText(activeAudioFileId))) return;
                setCopied(true);
                showToast("Audio ID copied");
                window.setTimeout(() => setCopied(false), 1200);
              }}
              className={cn(
                "inline-flex items-center gap-1 font-mono transition-all active:scale-95",
                copied ? "text-emerald-600" : "text-txt-dim hover:text-txt",
              )}
            >
              <Icon name={copied ? "check" : "copy"} size={11} strokeWidth={2.4} />
              {copied ? "Copied" : activeAudioFileId}
            </button>
          </div>
        </div>
        <span className={cn("flex items-center gap-1.5 text-xs font-semibold", dirty ? "text-amber-700" : "text-txt-mute")}>
          {dirty ? (
            <span className="h-[7px] w-[7px] rounded-full bg-amber-500" />
          ) : (
            <Icon name="check" size={14} strokeWidth={2.5} className="text-emerald-600" />
          )}
          {dirty ? "Unsaved changes" : "All changes saved"}
        </span>
        <Button
          variant={dirty ? "primary" : "ghost"}
          disabled={!dirty}
          onClick={() => { void saveSegments(); }}
        >
          Save
        </Button>
      </div>

      {/* player */}
      <div className="mb-4 rounded-[10px] border border-line bg-panel p-4">
        {/* ponytail: click-to-seek + zoom timeline with lane-stacked segments; drag-to-resize of blocks is deferred. */}
        <SegmentTimeline
          segs={segs}
          dur={dur}
          playPos={playPos}
          selId={segSel}
          viewStart={viewStart}
          viewEnd={viewEnd}
          seed={seed}
          onSeek={seek}
          onSelect={select}
          onSetView={setView}
          onSegTime={setSegTime}
          minimapPeaks={minimapWaveform.data?.peaks}
          viewPeaks={viewWaveform.data?.peaks}
        />
        <div className="mt-3.5 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <TBtn icon="skip-back" title="To start" onClick={() => seek(0)} />
            <TBtn icon="rewind" title="Back 1s" onClick={() => seek(playPos - 1)} />
            <button
              onClick={togglePlay}
              title="Play / pause"
              className="flex h-[46px] w-[46px] items-center justify-center rounded-full bg-blue-500 text-white hover:bg-blue-600"
            >
              <Icon name={playing ? "pause" : "play"} size={20} strokeWidth={2.2} />
            </button>
            <TBtn icon="rewind" title="Forward 1s" flip onClick={() => seek(playPos + 1)} />
            <TBtn icon="skip-fwd" title="To end" onClick={() => seek(dur)} />
          </div>
          <div className="min-w-[150px] font-mono text-sm font-semibold tabular-nums text-txt">
            {fmtClock(playPos)}
            <span className="text-txt-mute"> / {fmtClock(dur)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Icon name="gauge" size={15} strokeWidth={2} className="text-txt-mute" />
            <div className="relative">
              <select
                value={String(speed)}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="h-8 appearance-none rounded-md bg-panel-2 pl-2.5 pr-6 text-[12.5px] font-semibold tabular-nums text-txt outline-none"
              >
                {SPEEDS.map((v) => (
                  <option key={v} value={v}>
                    {v}×
                  </option>
                ))}
              </select>
              <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-txt-dim">
                <Icon name="chevron-down" size={12} strokeWidth={2.4} />
              </span>
            </div>
          </div>
          <div className="flex w-[150px] items-center gap-2">
            <Icon name="volume" size={15} strokeWidth={2} className="text-txt-mute" />
            <Slider value={volume} onChange={setVolume} min={0} max={1} step={0.01} format={(v) => `${Math.round(v * 100)}%`} />
          </div>
          <div className="flex-1" />
          <div className="flex items-center gap-1 rounded-md bg-panel-2 p-0.5">
            {abBtn("A", abA != null, setA)}
            {abBtn("B", abB != null, setB)}
            <button
              onClick={toggleLoop}
              title="Loop"
              className={cn("flex h-7 w-[30px] items-center justify-center rounded", loop ? "bg-blue-500 text-white" : "text-txt-dim")}
            >
              <Icon name="repeat" size={14} strokeWidth={2.2} />
            </button>
            {abA != null || abB != null ? (
              <button onClick={clearAB} title="Clear A-B" className="flex h-7 w-7 items-center justify-center rounded text-txt-mute">
                <Icon name="x" size={13} strokeWidth={2.4} />
              </button>
            ) : null}
          </div>
          <div className="flex items-center gap-1">
            <TBtn icon="zoom-out" title="Zoom out" onClick={zoomOut} />
            <TBtn icon="zoom-in" title="Zoom in" onClick={zoomIn} />
          </div>
        </div>
      </div>

      {/* segments */}
      <div className="flex min-h-0 flex-1 flex-col rounded-[10px] border border-line bg-panel p-4">
        <div className="mb-3 flex items-center gap-3">
          <div className="text-[15px] font-bold">
            Segments <span className="font-semibold text-txt-mute">{q ? `(${vis.length} of ${segs.length})` : `(${segs.length})`}</span>
          </div>
          <SearchInput value={segQuery} onChange={setQuery} placeholder="Search transcripts / phonemes…" />
          <div className="flex-1" />
          <Button variant="ghost" icon="plus" onClick={addSeg}>
            Add segment
          </Button>
        </div>
        {vis.length ? (
          <VirtualTable
            count={vis.length}
            estimateRowHeight={72}
            className="flex-1"
            renderRow={(i) => (
              <SegmentRow seg={vis[i]!} index={segs.indexOf(vis[i]!)} isLast={segs.indexOf(vis[i]!) === segs.length - 1} />
            )}
          />
        ) : (
          <div className="p-10 text-center text-[13px] text-txt-mute">
            {q ? `No segments match "${segQuery}".` : "No segments — add one to begin transcribing."}
          </div>
        )}
      </div>
    </div>
  );
}

function hashSeed(value: string): number {
  let out = 0;
  for (let i = 0; i < value.length; i += 1) out = (out * 31 + value.charCodeAt(i)) >>> 0;
  return out || 1;
}
