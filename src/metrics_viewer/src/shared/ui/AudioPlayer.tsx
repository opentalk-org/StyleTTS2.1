import { Pause, Play, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { cn } from "./cn";

export interface AudioPlayerProps {
  src: string;
  /** Accessible name, e.g. the run and artifact this clip belongs to. */
  label: string;
  className?: string;
}

const BARS = 120;
const VIEW_HEIGHT = 40;

/** Decoded peaks keyed by source, so stepping back and forth does not refetch. */
const peakCache = new Map<string, number[]>();
/** Runs comparing the same clip mount together; they should decode it once. */
const inFlight = new Map<string, Promise<number[]>>();
let sharedContext: AudioContext | null = null;

/**
 * Reduces a clip to one peak per bar. Decoding needs the whole file, which is fine
 * for short validation clips and is cached; anything that fails falls back to a
 * plain progress bar rather than blocking playback.
 */
function loadPeaks(src: string): Promise<number[]> {
  const cached = peakCache.get(src);
  if (cached !== undefined) return Promise.resolve(cached);
  const pending = inFlight.get(src);
  if (pending !== undefined) return pending;

  const request = decodePeaks(src).finally(() => inFlight.delete(src));
  inFlight.set(src, request);
  return request;
}

async function decodePeaks(src: string): Promise<number[]> {
  const response = await fetch(src);
  const buffer = await response.arrayBuffer();
  sharedContext ??= new AudioContext();
  const audio = await sharedContext.decodeAudioData(buffer);
  const samples = audio.getChannelData(0);
  const bucket = Math.floor(samples.length / BARS) || 1;

  const peaks: number[] = [];
  let loudest = 0;
  for (let bar = 0; bar < BARS; bar += 1) {
    let peak = 0;
    const start = bar * bucket;
    for (let index = start; index < start + bucket && index < samples.length; index += 1) {
      const value = Math.abs(samples[index]);
      if (value > peak) peak = value;
    }
    peaks.push(peak);
    if (peak > loudest) loudest = peak;
  }
  // Normalise so quiet clips still fill the strip.
  const normalised = loudest === 0 ? peaks : peaks.map((peak) => peak / loudest);
  peakCache.set(src, normalised);
  return normalised;
}

/** Compact transport with a waveform scrubber, built on a headless audio element. */
export function AudioPlayer({ src, label, className }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [failed, setFailed] = useState(false);
  const [peaks, setPeaks] = useState<number[] | null>(null);

  useEffect(() => {
    let current = true;
    setPeaks(peakCache.get(src) ?? null);
    loadPeaks(src)
      .then((values) => {
        if (current) setPeaks(values);
      })
      // A missing waveform is cosmetic; playback is unaffected.
      .catch(() => undefined);
    return () => {
      current = false;
    };
  }, [src]);

  function toggle() {
    const audio = audioRef.current;
    if (audio === null) return;
    if (audio.paused) void audio.play().catch(() => setFailed(true));
    else audio.pause();
  }

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (audio === null) return;
    audio.currentTime = seconds;
    setTime(seconds);
  }

  const progress = duration > 0 ? time / duration : 0;

  if (failed) {
    return (
      <p className={cn("m-0 flex items-center gap-2 px-3 py-4 text-xs text-fg-muted", className)}>
        <TriangleAlert size={13} className="shrink-0" />
        Audio could not be played
      </p>
    );
  }

  return (
    <div className={cn("flex items-center gap-2.5 px-3 py-3", className)}>
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
        onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onError={() => setFailed(true)}
      />
      <button
        type="button"
        aria-label={`${playing ? "Pause" : "Play"} ${label}`}
        onClick={toggle}
        className={cn(
          "grid size-7 shrink-0 place-items-center rounded-full border transition-colors duration-150 ease-out active:scale-[0.98]",
          playing
            ? "border-accent-border bg-accent-surface text-accent-bright"
            : "border-line-hover bg-inset text-fg-secondary hover:border-accent-border hover:text-fg",
        )}
      >
        {playing ? (
          <Pause size={12} fill="currentColor" />
        ) : (
          <Play size={12} fill="currentColor" className="ml-px" />
        )}
      </button>

      {/* The range sits transparently on top: it keeps native drag and arrow-key
          seeking while the waveform below does the drawing. */}
      <div className="relative h-8 min-w-0 flex-1">
        {peaks === null ? (
          <>
            <span aria-hidden className="absolute inset-x-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-surface-hover" />
            <span
              aria-hidden
              className="absolute left-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-accent"
              style={{ width: `${progress * 100}%` }}
            />
          </>
        ) : (
          <Waveform peaks={peaks} progress={progress} />
        )}
        <input
          type="range"
          aria-label={`Seek ${label}`}
          min={0}
          max={duration === 0 ? 1 : duration}
          step={0.01}
          value={time}
          disabled={duration === 0}
          onChange={(event) => seek(Number(event.target.value))}
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-default"
        />
      </div>

      <span className="shrink-0 font-mono text-[11px] tabular-nums text-fg-muted">
        {formatTime(time)} / {formatTime(duration)}
      </span>
    </div>
  );
}

function Waveform({ peaks, progress }: { peaks: number[]; progress: number }) {
  const played = progress * peaks.length;
  return (
    <svg
      aria-hidden
      viewBox={`0 0 ${peaks.length} ${VIEW_HEIGHT}`}
      preserveAspectRatio="none"
      className="absolute inset-0 h-full w-full"
    >
      {peaks.map((peak, index) => {
        const height = Math.max(1.5, peak * VIEW_HEIGHT);
        return (
          <rect
            key={index}
            x={index + 0.15}
            width={0.7}
            y={(VIEW_HEIGHT - height) / 2}
            height={height}
            rx={0.35}
            className={index < played ? "fill-accent-bright" : "fill-white/15"}
          />
        );
      })}
      {progress > 0 ? (
        <rect x={played} width={0.15} y={0} height={VIEW_HEIGHT} className="fill-accent" />
      ) : null}
    </svg>
  );
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
