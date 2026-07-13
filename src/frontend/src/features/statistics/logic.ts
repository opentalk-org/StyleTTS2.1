import type { BigramMatrix, Histogram, Pair, ScatterPoint, StatisticsPayload } from "./api";

export type Tone = "blue" | "emerald" | "amber" | "red";

export type HBarItem = { label: string; value: number; display: string };

export type RankItem = { label: string; value: number };

export type HistogramConfig = {
  title: string;
  unit: string;
  edges: number[];
  counts: number[];
  underflow: number;
  overflow: number;
  tone: Tone;
};

export type ScatterConfig = {
  title: string;
  unit: string;
  points: ScatterPoint[];
  xLabel: string;
  yLabel: string;
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

function histConfig(title: string, unit: string, h: Histogram, tone: Tone): HistogramConfig {
  return { title, unit, edges: h.edges, counts: h.counts, underflow: h.underflow ?? 0, overflow: h.overflow ?? 0, tone };
}

function hbars(pairs: Pair[], fmt: (v: number) => string, limit: number): HBarItem[] {
  return pairs.slice(0, limit).map(([label, value]) => ({ label, value, display: fmt(value) }));
}

function ranks(pairs: Pair[]): RankItem[] {
  return pairs.map(([label, value]) => ({ label, value }));
}

export function audioHistograms(p: StatisticsPayload): HistogramConfig[] {
  const configs = [histConfig("Duration per file", "seconds", p.duration_seconds_histogram, "blue")];
  if (!p.acoustic_metrics_available) return configs;
  configs.push(
    histConfig("Frame RMS", "dB", p.rms_db_histogram, "blue"),
    histConfig("Silence ratio", "ratio", p.silence_ratio_histogram, "amber"),
    histConfig("Mean RMS / file", "dB", p.mean_rms_nonsilent_db_per_file_histogram, "blue"),
    histConfig("Sample RMS / file", "dB", p.sample_rms_nonsilent_db_per_file_histogram, "blue"),
    histConfig("Silent-frame RMS", "dB", p.silence_rms_db_histogram, "amber"),
    histConfig("Frame min sample", "amplitude", p.frame_value_min_histogram, "emerald"),
    histConfig("Frame max sample", "amplitude", p.frame_value_max_histogram, "emerald"),
    histConfig("Frame mean sample", "amplitude", p.frame_value_mean_histogram, "emerald"),
  );
  // Only present when segments carry word-level alignment; otherwise the gap histogram is all
  // zeros and would read as a real (empty) distribution, so drop it entirely.
  const silence = p.inter_word_silence_seconds_histogram;
  if (silence && silence.counts.some((count) => count > 0)) {
    configs.push(histConfig("Silence between words", "seconds", silence, "amber"));
  }
  return configs;
}

// Per-sample speaking-rate scatters. Points are pre-sampled server-side; x is the clip
// duration so short clips with inflated rates are easy to spot.
export function rateScatters(p: StatisticsPayload): ScatterConfig[] {
  const wps = p.words_per_second_scatter ?? [];
  const cps = p.chars_per_second_scatter ?? [];
  // Rows carrying the total (length >= 3) support the count-axis projections; older 2-column
  // entries simply yield an empty count plot instead of NaN points.
  const withTotal = (rows: ScatterPoint[]) => rows.filter((row) => row.length >= 3);
  const wordsWithTotal = withTotal(wps);
  const charsWithTotal = withTotal(cps);
  return [
    { title: "Estimated words / second", unit: "vs duration", points: wps.map((r) => [r[0]!, r[1]!]), xLabel: "Duration (s)", yLabel: "Words / s", tone: "blue" },
    { title: "Estimated words / second", unit: "vs total words", points: wordsWithTotal.map((r) => [r[2]!, r[1]!]), xLabel: "Total words", yLabel: "Words / s", tone: "blue" },
    { title: "Estimated chars / second", unit: "vs duration", points: cps.map((r) => [r[0]!, r[1]!]), xLabel: "Duration (s)", yLabel: "Chars / s", tone: "emerald" },
    { title: "Estimated chars / second", unit: "vs total chars", points: charsWithTotal.map((r) => [r[2]!, r[1]!]), xLabel: "Total chars", yLabel: "Chars / s", tone: "emerald" },
    { title: "Total words", unit: "vs duration", points: wordsWithTotal.map((r) => [r[0]!, r[2]!]), xLabel: "Duration (s)", yLabel: "Words", tone: "blue" },
    { title: "Total characters", unit: "vs duration", points: charsWithTotal.map((r) => [r[0]!, r[2]!]), xLabel: "Duration (s)", yLabel: "Characters", tone: "emerald" },
  ];
}

export function voiceHistograms(p: StatisticsPayload): HistogramConfig[] {
  return [
    histConfig("Samples per voice", "samples", p.voice_sample_count_histogram, "emerald"),
    histConfig("Duration per voice", "seconds", p.voice_duration_seconds_histogram, "blue"),
  ];
}

export type StatTileData = { label: string; value: string; sub?: string; tone: Tone };

export function audioTiles(p: StatisticsPayload): StatTileData[] {
  const clippedPct = p.file_count ? ((p.clipped_audio_file_count / p.file_count) * 100).toFixed(1) : "0.0";
  const avg = p.file_count ? p.total_duration_seconds / p.file_count : 0;
  const tiles: StatTileData[] = [
    { label: "Total duration", value: fmtDuration(p.total_duration_seconds), sub: `avg ${fmtDuration(avg)} / file`, tone: "blue" },
    { label: "Speakers", value: `${p.speaker_count}`, sub: `${p.segment_count} segments`, tone: "emerald" },
    {
      label: "Duplicate segments collapsed",
      value: fmtCompact(p.duplicate_segments_collapsed),
      sub: "multi-model copies merged",
      tone: p.duplicate_segments_collapsed > 0 ? "amber" : "blue",
    },
  ];
  if (p.acoustic_metrics_available) {
    tiles.unshift({
      label: "Clipped files",
      value: `${p.clipped_audio_file_count}`,
      sub: `of ${p.file_count} (${clippedPct}%)`,
      tone: p.clipped_audio_file_count > 0 ? "amber" : "blue",
    });
  }
  return tiles;
}

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
  lengthEdges: number[];
  lengthCounts: number[];
  lengthUnderflow: number;
  lengthOverflow: number;
  speakerLength: HBarItem[];
  grams1: HBarItem[];
  bigramMatrix: BigramMatrix;
  trigramsTop: RankItem[];
  trigramsBottom: RankItem[];
};

export function corpusData(p: StatisticsPayload, tab: CorpusTab): CorpusData {
  const isIpa = tab === "ipa";
  const lengthHist = isIpa ? p.phoneme_count_per_file_histogram : p.char_count_per_file_histogram;
  const grams = isIpa ? p.phoneme_unigram_counts : p.char_unigram_counts;
  const bigramMatrix = isIpa ? p.phoneme_bigram_matrix : p.char_bigram_matrix;
  const speaker = isIpa ? p.speaker_phoneme_count : p.speaker_char_count;
  const top = isIpa ? p.phoneme_trigram_top10 : p.char_trigram_top10;
  const bottom = isIpa ? p.phoneme_trigram_bottom10 : p.char_trigram_bottom10;
  return {
    unit: isIpa ? "phonemes" : "characters",
    available: isIpa ? p.phonemes_available : true,
    lengthEdges: lengthHist.edges,
    lengthCounts: lengthHist.counts,
    lengthUnderflow: lengthHist.underflow ?? 0,
    lengthOverflow: lengthHist.overflow ?? 0,
    speakerLength: hbars(speaker, fmtCompact, 15),
    grams1: hbars(grams, fmtCompact, 24),
    bigramMatrix,
    trigramsTop: ranks(top),
    trigramsBottom: ranks(bottom),
  };
}
