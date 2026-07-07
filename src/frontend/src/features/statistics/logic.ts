/** Chart accent, mapped to Tailwind classes by the chart components. */
export type Tone = "blue" | "emerald" | "amber" | "red";

/** A horizontal bar row (per-speaker totals, 1-gram frequency). */
export type HBarItem = { label: string; value: number; display: string };

/** A ranked-list row (trigrams). Rendered with an index + value bar. */
export type RankItem = { label: string; value: number };

/** One of the nine audio distribution histograms. */
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

/** Deterministic 0-1 pseudo-random from an integer seed. */
function rng(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

/**
 * Deterministic gaussian-ish histogram bins, normalized to [0,1]. `skew` shifts
 * the peak along the x-axis; `seed` decorrelates the per-bin jitter so each
 * distribution looks distinct while staying stable across renders.
 */
export function gaussianBins(n: number, seed: number, skew: number): number[] {
  const raw: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i / n - 0.5) * 2;
    const jitter = 0.7 + 0.3 * rng(seed * 50 + i);
    raw.push(Math.exp(-Math.pow((x - skew) * 2, 2)) * jitter);
  }
  const max = Math.max(...raw);
  return raw.map((v) => v / max);
}

/** The nine audio distributions shown in the Audio section. */
export const AUDIO_HISTOGRAMS: HistogramConfig[] = [
  { title: "Clip duration", unit: "seconds", bins: gaussianBins(24, 3, -0.1), xmin: "0s", xmid: "12s", xmax: "30s", tone: "blue" },
  { title: "Frame RMS", unit: "dB", bins: gaussianBins(24, 7, 0.2), xmin: "-60", xmid: "-30", xmax: "0", tone: "blue" },
  { title: "Frame min sample", unit: "amplitude", bins: gaussianBins(24, 11, -0.3), xmin: "-1.0", xmid: "-0.5", xmax: "0", tone: "emerald" },
  { title: "Frame max sample", unit: "amplitude", bins: gaussianBins(24, 13, 0.3), xmin: "0", xmid: "0.5", xmax: "1.0", tone: "emerald" },
  { title: "Frame mean sample", unit: "amplitude", bins: gaussianBins(24, 17, 0), xmin: "-0.1", xmid: "0", xmax: "0.1", tone: "emerald" },
  { title: "Mean RMS / file", unit: "dB", bins: gaussianBins(24, 19, 0.1), xmin: "-50", xmid: "-25", xmax: "0", tone: "blue" },
  { title: "Sample RMS / file", unit: "dB", bins: gaussianBins(24, 23, 0.15), xmin: "-50", xmid: "-25", xmax: "0", tone: "blue" },
  { title: "Silence ratio", unit: "ratio", bins: gaussianBins(24, 29, -0.35), xmin: "0", xmid: "0.5", xmax: "1.0", tone: "amber" },
  { title: "Silent-frame RMS", unit: "dB", bins: gaussianBins(24, 31, -0.2), xmin: "-80", xmid: "-50", xmax: "-20", tone: "amber" },
];

/** Files that fall outside the trainable text-length range. */
export const TEXT_WARNINGS = [
  { file: "take_138.wav", why: "Transcript too long", detail: "412 chars · max 300" },
  { file: "line_092.wav", why: "Phoneme string too short", detail: "2 symbols · min 5" },
  { file: "session_117.wav", why: "Empty transcript", detail: "0 chars" },
];

export const SPEAKER_DURATION: HBarItem[] = [
  { label: "Maya Chen", value: 4.2, display: "4.2 h" },
  { label: "Theo Park", value: 3.1, display: "3.1 h" },
  { label: "Aria Russo", value: 2.6, display: "2.6 h" },
  { label: "Sam Okafor", value: 1.4, display: "1.4 h" },
  { label: "Noah Vance", value: 0.9, display: "0.9 h" },
];

const SPEAKER_LENGTH: HBarItem[] = [
  { label: "Maya Chen", value: 182000, display: "182k" },
  { label: "Theo Park", value: 131000, display: "131k" },
  { label: "Aria Russo", value: 108000, display: "108k" },
  { label: "Sam Okafor", value: 61000, display: "61k" },
  { label: "Noah Vance", value: 39000, display: "39k" },
];

const GRAMS_TRANSCRIPT: [string, number][] = [["␣", 41200], ["e", 38900], ["t", 29100], ["a", 25600], ["o", 24300], ["n", 22800], ["i", 21900], ["s", 20400], ["r", 19800], ["h", 18200], ["l", 13900], ["d", 12600], ["c", 9800], ["u", 8700], ["m", 8100]];
const GRAMS_IPA: [string, number][] = [["ə", 8120], ["t", 6890], ["n", 5980], ["s", 5410], ["ɪ", 5120], ["l", 4760], ["ɹ", 4530], ["d", 4210], ["k", 3980], ["m", 3640], ["ɛ", 3410], ["oʊ", 3120], ["i", 2980], ["z", 2740], ["w", 2510]];

const TRIGRAMS_TRANSCRIPT = {
  top: [["the", 4120], [" an", 3880], ["ing", 3560], ["nd ", 3310], ["ed ", 3090], [" th", 2880], ["he ", 2640], [" to ", 2410], ["er ", 2280], ["at ", 2140]] as [string, number][],
  bot: [["qzx", 2], ["jvw", 3], ["zqu", 4], ["xwy", 5], ["vkx", 6], ["jqz", 7], ["wqx", 8], ["qjp", 9], ["zxc", 10], ["vqj", 11]] as [string, number][],
};
const TRIGRAMS_IPA = {
  top: [["ðiː", 412], ["ænd", 388], ["ɪŋɡ", 356], ["ɪzə", 331], ["tuː", 309], ["fɔːɹ", 288], ["ðæts", 264], ["wɪð", 241], ["hiːz", 228], ["ɑːɹ", 214]] as [string, number][],
  bot: [["ʒuː", 3], ["ŋkt", 4], ["ðʒə", 5], ["pʍ", 6], ["tsk", 7], ["ʃtʃ", 8], ["ksθ", 9], ["mbz", 10], ["ŋɡθ", 11], ["dʒd", 12]] as [string, number][],
};

export type CorpusTab = "transcript" | "ipa";

export type CorpusData = {
  unit: string;
  lengthBins: number[];
  lengthAxis: { xmin: string; xmid: string; xmax: string };
  speakerLength: HBarItem[];
  grams1: HBarItem[];
  trigramsTop: RankItem[];
  trigramsBottom: RankItem[];
};

/** Text-corpus charts derived from the active tab (raw transcript vs. IPA). */
export function corpusData(tab: CorpusTab): CorpusData {
  const isIpa = tab === "ipa";
  const grams = isIpa ? GRAMS_IPA : GRAMS_TRANSCRIPT;
  const trigrams = isIpa ? TRIGRAMS_IPA : TRIGRAMS_TRANSCRIPT;
  return {
    unit: isIpa ? "phonemes" : "characters",
    lengthBins: gaussianBins(26, 41, isIpa ? 0.1 : 0.05),
    lengthAxis: { xmin: "0", xmid: isIpa ? "150" : "180", xmax: isIpa ? "400" : "500" },
    speakerLength: SPEAKER_LENGTH,
    grams1: grams.map(([label, value]) => ({
      label: label === "␣" ? "(space)" : label,
      value,
      display: value.toLocaleString(),
    })),
    trigramsTop: trigrams.top.map(([label, value]) => ({ label, value })),
    trigramsBottom: trigrams.bot.map(([label, value]) => ({ label, value })),
  };
}
