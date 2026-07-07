import type { Histogram, Pair, StatisticsPayload } from "./api";

/** Chart accent, mapped to Tailwind classes by the chart components. */
export type Tone = "blue" | "emerald" | "amber" | "red";

/** A horizontal bar row (per-speaker totals, 1-gram frequency). */
export type HBarItem = { label: string; value: number; display: string };

/** A ranked-list row (trigrams). Rendered with an index + value bar. */
export type RankItem = { label: string; value: number };

/** One of the audio distribution histograms. */
export type HistogramConfig = {
  title: string;
  unit: string;
  bins: number[];
  xmin: string;
  xmid: string;
  xmax: string;
  tone: Tone;
};

export const BAR_CLASS: Record<Tone, string> = {
  blue: "bg-blue-500",
  emerald: "bg-emerald-500",
  amber: "bg-amber-500",
  red: "bg-red-500",
};

export const TILE_TEXT_CLASS: Record<Tone, string> = {
  blue: "text-blue-600",
  emerald: "text-emerald-700",
  amber: "text-amber-700",
  red: "text-red-600",
};

const fmtDb = (v: number) => `${v.toFixed(0)}`;
const fmtAmp = (v: number) => v.toFixed(2);
const fmtRatio = (v: number) => v.toFixed(2);
const fmtSeconds = (v: number) => (v < 10 ? `${v.toFixed(1)}s` : `${Math.round(v)}s`);
const fmtInt = (v: number) => `${Math.round(v)}`;

function fmtCompact(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(v >= 10_000 ? 0 : 1)}k`;
  return `${Math.round(v)}`;
}

export function fmtDuration(seconds: number): string {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }
  return `${seconds.toFixed(1)}s`;
}

function histAxis(h: Histogram, fmt: (v: number) => string): { xmin: string; xmid: string; xmax: string } {
  const edges = h.edges.length ? h.edges : [0, 1];
  return {
    xmin: fmt(edges[0]!),
    xmid: fmt(edges[Math.floor(edges.length / 2)]!),
    xmax: fmt(edges[edges.length - 1]!),
  };
}

function histConfig(title: string, unit: string, h: Histogram, tone: Tone, fmt: (v: number) => string): HistogramConfig {
  return { title, unit, bins: h.counts, ...histAxis(h, fmt), tone };
}

function hbars(pairs: Pair[], fmt: (v: number) => string, limit: number): HBarItem[] {
  return pairs.slice(0, limit).map(([label, value]) => ({ label, value, display: fmt(value) }));
}

function ranks(pairs: Pair[]): RankItem[] {
  return pairs.map(([label, value]) => ({ label, value }));
}

/** The audio-signal distributions shown in the Audio section. */
export function audioHistograms(p: StatisticsPayload): HistogramConfig[] {
  return [
    histConfig("Duration per file", "seconds", p.duration_seconds_histogram, "blue", fmtSeconds),
    histConfig("Frame RMS", "dB", p.rms_db_histogram, "blue", fmtDb),
    histConfig("Silence ratio", "ratio", p.silence_ratio_histogram, "amber", fmtRatio),
    histConfig("Mean RMS / file", "dB", p.mean_rms_nonsilent_db_per_file_histogram, "blue", fmtDb),
    histConfig("Sample RMS / file", "dB", p.sample_rms_nonsilent_db_per_file_histogram, "blue", fmtDb),
    histConfig("Silent-frame RMS", "dB", p.silence_rms_db_histogram, "amber", fmtDb),
    histConfig("Frame min sample", "amplitude", p.frame_value_min_histogram, "emerald", fmtAmp),
    histConfig("Frame max sample", "amplitude", p.frame_value_max_histogram, "emerald", fmtAmp),
    histConfig("Frame mean sample", "amplitude", p.frame_value_mean_histogram, "emerald", fmtAmp),
  ];
}

export type StatTileData = { label: string; value: string; sub?: string; tone: Tone };

/** Headline scalar tiles for the Audio section. */
export function audioTiles(p: StatisticsPayload): StatTileData[] {
  const clippedPct = p.file_count ? ((p.clipped_audio_file_count / p.file_count) * 100).toFixed(1) : "0.0";
  const avg = p.file_count ? p.total_duration_seconds / p.file_count : 0;
  return [
    {
      label: "Clipped files",
      value: `${p.clipped_audio_file_count}`,
      sub: `of ${p.file_count} (${clippedPct}%)`,
      tone: p.clipped_audio_file_count > 0 ? "amber" : "blue",
    },
    { label: "Total duration", value: fmtDuration(p.total_duration_seconds), sub: `avg ${fmtDuration(avg)} / file`, tone: "blue" },
    { label: "Speakers", value: `${p.speaker_count}`, sub: `${p.segment_count} segments`, tone: "emerald" },
    {
      label: "Duplicate segments collapsed",
      value: fmtCompact(p.duplicate_segments_collapsed),
      sub: "multi-model copies merged",
      tone: p.duplicate_segments_collapsed > 0 ? "amber" : "blue",
    },
  ];
}

/** Per-speaker duration bars (hours), most-talkative first, capped. */
export function speakerDuration(p: StatisticsPayload): HBarItem[] {
  return hbars(p.speaker_duration_seconds, (v) => fmtDuration(v), 15);
}

/** Files outside the trainable text-length range (for the warning banner). */
export function textWarnings(p: StatisticsPayload): { file: string; why: string; detail: string }[] {
  const reasons: Record<string, string> = {
    "empty transcript": "Empty transcript",
    "too short": "Transcript too short",
    "too long": "Transcript too long",
  };
  const min = p.params.text_min_chars ?? 5;
  const max = p.params.text_max_chars ?? 500;
  return p.text_length_warnings.map((w) => ({
    file: w.name,
    why: reasons[w.reason] ?? w.reason,
    detail:
      w.reason === "too long"
        ? `${w.char_count} chars · max ${max}`
        : w.reason === "empty transcript"
          ? "0 chars"
          : `${w.char_count} chars · min ${min}`,
  }));
}

export type CorpusTab = "transcript" | "ipa";

export type CorpusData = {
  unit: string;
  available: boolean;
  lengthBins: number[];
  lengthAxis: { xmin: string; xmid: string; xmax: string };
  speakerLength: HBarItem[];
  grams1: HBarItem[];
  trigramsTop: RankItem[];
  trigramsBottom: RankItem[];
};

/** Text-corpus charts derived from the active tab (raw transcript vs. IPA). */
export function corpusData(p: StatisticsPayload, tab: CorpusTab): CorpusData {
  const isIpa = tab === "ipa";
  const lengthHist = isIpa ? p.phoneme_count_per_file_histogram : p.char_count_per_file_histogram;
  const grams = isIpa ? p.phoneme_unigram_counts : p.char_unigram_counts;
  const speaker = isIpa ? p.speaker_phoneme_count : p.speaker_char_count;
  const top = isIpa ? p.phoneme_trigram_top10 : p.char_trigram_top10;
  const bottom = isIpa ? p.phoneme_trigram_bottom10 : p.char_trigram_bottom10;
  return {
    unit: isIpa ? "phonemes" : "characters",
    available: isIpa ? p.phonemes_available : true,
    lengthBins: lengthHist.counts,
    lengthAxis: histAxis(lengthHist, fmtInt),
    speakerLength: hbars(speaker, fmtCompact, 15),
    grams1: hbars(grams, fmtCompact, 24),
    trigramsTop: ranks(top),
    trigramsBottom: ranks(bottom),
  };
}
