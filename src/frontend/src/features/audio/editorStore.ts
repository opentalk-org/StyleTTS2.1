import { create } from "zustand";

import type { Segment } from "./api";

const MIN_SPAN = 2;
/** Default window span on load — start zoomed in so long files stay light. */
const DEFAULT_SPAN = 45;

/** Keep segments ordered by start time — the list and timeline lanes rely on it. */
function sortSegs(segs: Segment[]): Segment[] {
  return [...segs].sort((a, b) => a.start - b.start || a.end - b.end);
}

function clampView(start: number, end: number, dur: number): { viewStart: number; viewEnd: number } {
  const span = Math.min(dur, Math.max(MIN_SPAN, end - start));
  const viewStart = Math.max(0, Math.min(start, dur - span));
  return { viewStart, viewEnd: viewStart + span };
}

type EditorStore = {
  /** File whose segments are currently loaded (guards re-loading). */
  fileId: string | null;
  dur: number;
  segs: Segment[];
  playPos: number;
  playing: boolean;
  speed: number;
  volume: number;
  loop: boolean;
  /** Visible slice of the audio [viewStart, viewEnd] — the timeline renders only this. */
  viewStart: number;
  viewEnd: number;
  dirty: boolean;
  segSel: string | null;
  /** Segment ids ticked for bulk actions (delete). Independent of `segSel`. */
  segChecked: string[];
  segQuery: string;
  load: (fileId: string, dur: number, segs: Segment[]) => void;
  seek: (playPos: number) => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  setVolume: (volume: number) => void;
  toggleLoop: () => void;
  setView: (start: number, end: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  followPlayhead: () => void;
  setQuery: (segQuery: string) => void;
  select: (segSel: string) => void;
  toggleCheck: (id: string) => void;
  setChecked: (ids: string[]) => void;
  deleteChecked: () => void;
  setSegTime: (id: string, start: number, end: number) => void;
  setSegText: (id: string, text: string) => void;
  setSegPhon: (id: string, phon: string) => void;
  setSegVoice: (id: string, speaker: string) => void;
  deleteSeg: (id: string) => void;
  mergeNext: (id: string) => void;
  addSeg: () => void;
  save: () => void;
};

function zoom(s: EditorStore, factor: number): { viewStart: number; viewEnd: number } {
  const span = s.viewEnd - s.viewStart;
  const inView = s.playPos >= s.viewStart && s.playPos <= s.viewEnd;
  const center = inView ? s.playPos : (s.viewStart + s.viewEnd) / 2;
  const next = Math.min(s.dur, Math.max(MIN_SPAN, span * factor));
  return clampView(center - next / 2, center + next / 2, s.dur);
}

export const useEditor = create<EditorStore>((set) => ({
  fileId: null,
  dur: 0,
  segs: [],
  playPos: 0,
  playing: false,
  speed: 1,
  volume: 1,
  loop: false,
  viewStart: 0,
  viewEnd: 0,
  dirty: false,
  segSel: null,
  segChecked: [],
  segQuery: "",
  load: (fileId, dur, segs) =>
    set({
      fileId, dur, segs: sortSegs(segs), playPos: 0, playing: false, dirty: false,
      segSel: null, segChecked: [], segQuery: "", loop: false,
      viewStart: 0, viewEnd: Math.min(dur, DEFAULT_SPAN),
    }),
  seek: (playPos) => set((s) => ({ playPos: Math.max(0, Math.min(s.dur, playPos)) })),
  togglePlay: () => set((s) => ({ playing: !s.playing })),
  setSpeed: (speed) => set({ speed }),
  setVolume: (volume) => set({ volume }),
  toggleLoop: () => set((s) => ({ loop: !s.loop })),
  setView: (start, end) => set((s) => clampView(start, end, s.dur)),
  zoomIn: () => set((s) => zoom(s, 1 / 2)),
  zoomOut: () => set((s) => zoom(s, 2)),
  followPlayhead: () =>
    set((s) => {
      if (s.playPos >= s.viewStart && s.playPos <= s.viewEnd) return s;
      const span = s.viewEnd - s.viewStart;
      return clampView(s.playPos - span / 2, s.playPos + span / 2, s.dur);
    }),
  setQuery: (segQuery) => set({ segQuery }),
  select: (segSel) => set({ segSel }),
  toggleCheck: (id) =>
    set((s) => ({
      segChecked: s.segChecked.includes(id) ? s.segChecked.filter((x) => x !== id) : [...s.segChecked, id],
    })),
  setChecked: (ids) => set({ segChecked: ids }),
  deleteChecked: () =>
    set((s) => {
      if (!s.segChecked.length) return s;
      const drop = new Set(s.segChecked);
      return { segs: s.segs.filter((g) => !drop.has(g.id)), segChecked: [], dirty: true };
    }),
  setSegTime: (id, start, end) =>
    set((s) => {
      if (end - start < 0.1) return s;
      const ns = Math.max(0, start);
      const ne = Math.min(s.dur, end);
      return { segs: s.segs.map((g) => (g.id === id ? { ...g, start: ns, end: ne } : g)), dirty: true };
    }),
  setSegText: (id, text) =>
    // Editing the text invalidates the word alignment produced for the old text.
    set((s) => ({ segs: s.segs.map((g) => (g.id === id ? { ...g, text, alignment: null } : g)), dirty: true })),
  setSegPhon: (id, phon) =>
    set((s) => ({ segs: s.segs.map((g) => (g.id === id ? { ...g, phon } : g)), dirty: true })),
  setSegVoice: (id, speaker) =>
    set((s) => ({ segs: s.segs.map((g) => (g.id === id ? { ...g, speaker } : g)), dirty: true })),
  deleteSeg: (id) =>
    set((s) => ({ segs: s.segs.filter((g) => g.id !== id), segChecked: s.segChecked.filter((x) => x !== id), dirty: true })),
  mergeNext: (id) =>
    set((s) => {
      const i = s.segs.findIndex((g) => g.id === id);
      if (i < 0 || i >= s.segs.length - 1) return s;
      const cur = s.segs[i]!;
      const next = s.segs[i + 1]!;
      const mergedAlignment = cur.alignment || next.alignment ? [...(cur.alignment ?? []), ...(next.alignment ?? [])] : null;
      const merged: Segment = { ...cur, end: next.end, text: `${cur.text} ${next.text}`.trim(), phon: `${cur.phon} ${next.phon}`.trim(), alignment: mergedAlignment };
      const segs = [...s.segs];
      segs.splice(i, 2, merged);
      return { segs, dirty: true, segSel: merged.id };
    }),
  addSeg: () =>
    set((s) => {
      const start = s.playPos;
      const seg: Segment = { id: `seg_${Date.now()}`, start, end: Math.min(s.dur, start + 2), text: "", phon: "", speaker: "", type_: "manual" };
      return { segs: sortSegs([...s.segs, seg]), dirty: true, segSel: seg.id };
    }),
  save: () => set({ dirty: false }),
}));
