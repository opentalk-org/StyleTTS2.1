import { type PointerEvent, useRef } from "react";

import { fmtDur } from "@/shared/format";
import { WaveformBars } from "@/shared/media/WaveformBars";
import { cn } from "@/shared/ui/cn";
import type { Segment } from "./api";
import { WaveformPeaks } from "./WaveformPeaks";

const LANE_H = 30;
const MIN_BODY = 96;
const MAX_BODY = 240;
const MAX_RENDER_LANES = 8;
const MIN_SEG = 0.1;
const TICK_TARGETS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900];

type DragMode = "move" | "l" | "r";
type Drag = { id: string; mode: DragMode; startX: number; s0: number; e0: number; pxPerSec: number; moved: boolean };

/**
 * Greedy interval-graph lane assignment: each segment goes in the first lane
 * whose last segment has already ended, else a new lane. Overlapping segments
 * land in different lanes. Expects `segs` pre-sorted by start.
 */
function assignLanes(segs: Segment[]): { placed: { seg: Segment; lane: number }[]; lanes: number } {
  const laneEnds: number[] = [];
  const placed = segs.map((seg) => {
    let lane = laneEnds.findIndex((end) => seg.start >= end - 1e-6);
    if (lane === -1) {
      lane = laneEnds.length;
      laneEnds.push(seg.end);
    } else {
      laneEnds[lane] = Math.max(laneEnds[lane]!, seg.end);
    }
    return { seg, lane };
  });
  return { placed, lanes: Math.max(1, laneEnds.length) };
}

function ticks(start: number, end: number): number[] {
  const span = end - start;
  const step = TICK_TARGETS.find((t) => t >= span / 6) ?? 900;
  const out: number[] = [];
  for (let t = Math.ceil(start / step) * step; t <= end; t += step) out.push(t);
  return out;
}

export function SegmentTimeline({
  segs,
  dur,
  playPos,
  selId,
  viewStart,
  viewEnd,
  seed,
  onSeek,
  onSelect,
  onSetView,
  onSegTime,
  minimapPeaks,
  viewPeaks,
}: {
  segs: Segment[];
  dur: number;
  playPos: number;
  selId: string | null;
  viewStart: number;
  viewEnd: number;
  seed: number;
  onSeek: (t: number) => void;
  onSelect: (id: string) => void;
  onSetView: (start: number, end: number) => void;
  onSegTime: (id: string, start: number, end: number) => void;
  minimapPeaks?: [number, number][];
  viewPeaks?: [number, number][];
}) {
  const panning = useRef(false);
  const drag = useRef<Drag | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const span = Math.max(0.001, viewEnd - viewStart);
  const indexOf = new Map(segs.map((g, i) => [g.id, i]));

  const inView = segs.filter((g) => g.end > viewStart && g.start < viewEnd);
  const sorted = [...inView].sort((a, b) => a.start - b.start);
  const laneLayout = assignLanes(sorted);
  const tooDense = inView.length > 200 || laneLayout.lanes > MAX_RENDER_LANES;
  const { placed, lanes } = tooDense ? { placed: [], lanes: 1 } : laneLayout;
  const bodyH = Math.min(MAX_BODY, Math.max(MIN_BODY, lanes * LANE_H));
  const laneH = bodyH / lanes;
  const pct = (t: number) => ((t - viewStart) / span) * 100;

  // Per-word alignment marks for segments in view: a thin tick at each word start,
  // and a highlight band for the word the playhead currently sits in.
  const wordMarks = tooDense
    ? []
    : inView
        .flatMap((g) => g.alignment ?? [])
        .filter((w) => w.end > viewStart && w.start < viewEnd);
  const currentWord = wordMarks.find((w) => playPos >= w.start && playPos <= w.end) ?? null;

  const panTo = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const center = ((e.clientX - r.left) / r.width) * dur;
    onSetView(center - span / 2, center + span / 2);
  };
  const seekAt = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(viewStart + ((e.clientX - r.left) / r.width) * span);
  };

  const onBlockDown = (e: PointerEvent<HTMLDivElement>, seg: Segment) => {
    e.stopPropagation();
    const width = bodyRef.current?.getBoundingClientRect().width ?? 1;
    const mode = ((e.target as HTMLElement).dataset.handle as DragMode) ?? "move";
    drag.current = { id: seg.id, mode, startX: e.clientX, s0: seg.start, e0: seg.end, pxPerSec: width / span, moved: false };
    e.currentTarget.setPointerCapture(e.pointerId);
    onSelect(seg.id);
  };
  const onBlockMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d) return;
    const dt = (e.clientX - d.startX) / d.pxPerSec;
    if (Math.abs(e.clientX - d.startX) > 2) d.moved = true;
    if (d.mode === "move") {
      const len = d.e0 - d.s0;
      let s = Math.max(0, Math.min(dur - len, d.s0 + dt));
      onSegTime(d.id, s, s + len);
    } else if (d.mode === "l") {
      onSegTime(d.id, Math.min(d.e0 - MIN_SEG, Math.max(0, d.s0 + dt)), d.e0);
    } else {
      onSegTime(d.id, d.s0, Math.max(d.s0 + MIN_SEG, Math.min(dur, d.e0 + dt)));
    }
  };
  const onBlockUp = (seg: Segment) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) onSeek(seg.start);
  };

  return (
    <div>
      <div
        className="relative mb-2 h-9 cursor-pointer overflow-hidden rounded-md bg-panel-2"
        onPointerDown={(e) => { panning.current = true; e.currentTarget.setPointerCapture(e.pointerId); panTo(e); }}
        onPointerMove={(e) => { if (panning.current) panTo(e); }}
        onPointerUp={() => { panning.current = false; }}
        onLostPointerCapture={() => { panning.current = false; }}
      >
        <div className="pointer-events-none absolute inset-0 px-px text-blue-500 opacity-50">
          {minimapPeaks?.length ? <WaveformPeaks peaks={minimapPeaks} height={36} /> : <WaveformBars seed={seed} bars={160} height={36} />}
        </div>
        <div
          className="pointer-events-none absolute top-0 bottom-0 rounded border-[1.5px] border-blue-500 bg-blue-500/15"
          style={{ left: `${(viewStart / dur) * 100}%`, width: `${(span / dur) * 100}%` }}
        />
        <div className="pointer-events-none absolute top-0 bottom-0 w-px bg-gray-900" style={{ left: `${(playPos / dur) * 100}%` }} />
      </div>

      <div className="overflow-hidden rounded-lg bg-panel-2">
        <div className="relative h-5 border-b border-line">
          {ticks(viewStart, viewEnd).map((t) => (
            <div key={t} className="absolute top-0 bottom-0 border-l border-line" style={{ left: `${pct(t)}%` }}>
              <span className="absolute left-1 top-0.5 whitespace-nowrap font-mono text-[9.5px] tabular-nums text-txt-mute">
                {fmtDur(t)}
              </span>
            </div>
          ))}
        </div>

        <div ref={bodyRef} className="relative cursor-text" style={{ height: bodyH }} onPointerDown={seekAt}>
          <div className="pointer-events-none absolute inset-0 px-px text-blue-500 opacity-60">
            {viewPeaks?.length ? <WaveformPeaks peaks={viewPeaks} height={bodyH} /> : <WaveformBars seed={seed + Math.floor(viewStart)} bars={120} height={bodyH} />}
          </div>

          {placed.map(({ seg, lane }) => {
            const sel = seg.id === selId;
            const left = Math.max(0, pct(seg.start));
            const right = Math.min(100, pct(seg.end));
            return (
              <div
                key={seg.id}
                title={`#${(indexOf.get(seg.id) ?? 0) + 1} · ${seg.speaker}\n${seg.text}`}
                onPointerDown={(e) => onBlockDown(e, seg)}
                onPointerMove={onBlockMove}
                onPointerUp={() => onBlockUp(seg)}
                className={cn(
                  "group absolute cursor-grab touch-none overflow-hidden rounded-[3px] border text-[9px] font-bold active:cursor-grabbing",
                  sel ? "border-blue-600 bg-blue-500/25 text-blue-700" : "border-blue-500/50 bg-blue-500/10 text-blue-700 hover:bg-blue-500/20",
                )}
                style={{ left: `${left}%`, width: `max(6px, ${right - left}%)`, top: lane * laneH + 1, height: laneH - 2 }}
              >
                <span className="pointer-events-none absolute left-1.5 top-0.5 tabular-nums">{(indexOf.get(seg.id) ?? 0) + 1}</span>
                <div data-handle="l" className="absolute left-0 top-0 bottom-0 w-1.5 cursor-ew-resize bg-blue-600/0 group-hover:bg-blue-600/40" />
                <div data-handle="r" className="absolute right-0 top-0 bottom-0 w-1.5 cursor-ew-resize bg-blue-600/0 group-hover:bg-blue-600/40" />
              </div>
            );
          })}

          {currentWord ? (
            <div
              className="pointer-events-none absolute top-0 bottom-0 bg-amber-400/20"
              style={{ left: `${pct(currentWord.start)}%`, width: `max(2px, ${pct(currentWord.end) - pct(currentWord.start)}%)` }}
            />
          ) : null}

          {wordMarks.map((w, i) => (
            <div
              key={`${w.start}-${i}`}
              className={cn("pointer-events-none absolute top-0 bottom-0 w-px", w === currentWord ? "bg-amber-500/80" : "bg-blue-700/40")}
              style={{ left: `${pct(w.start)}%` }}
            />
          ))}

          {tooDense ? (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <span className="rounded-full bg-panel/80 px-3 py-1 text-[11px] font-semibold text-txt-dim">
                {inView.length} segments / {laneLayout.lanes} lanes in view - zoom in to edit on the timeline
              </span>
            </div>
          ) : null}

          {playPos >= viewStart && playPos <= viewEnd ? (
            <div className="pointer-events-none absolute top-0 bottom-0 w-0.5 bg-gray-900" style={{ left: `${pct(playPos)}%` }}>
              <div className="absolute -left-[3px] top-0 h-2.5 w-2.5 rounded-full bg-gray-900" />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
