import { rng } from "@/mock/constants";
import type { Checkpoint } from "@/mock/types";
import type { Option } from "@/shared/ui/Select";

const IPA_POOL =
  "ə t n s ɪ l ɹ k d m ɛ oʊ i z w æ aɪ eɪ ʃ θ ð v f p b ɡ".split(" ");

/** Mock grapheme→IPA: deterministic per word, plausible-looking phoneme string. */
export function phonemize(text: string): string {
  return text
    .trim()
    .split(/\s+/)
    .map((w, wi) => {
      const len = Math.max(2, Math.round(w.replace(/[^a-z]/gi, "").length * 0.85));
      let o = "";
      for (let c = 0; c < len; c++) {
        o += IPA_POOL[(w.charCodeAt(c % w.length) + c * 3 + wi) % IPA_POOL.length];
      }
      return o;
    })
    .join(" ");
}

/** Deterministic synthesis duration in seconds, seeded by a result id. */
export function synthDuration(id: string, salt = 0): number {
  return 2.2 + rng(id.length + salt) * 4;
}

/** Checkpoint dropdown options, StyleTTS2 only, with an empty prompt entry. */
export function checkpointOptions(checkpoints: Checkpoint[]): Option[] {
  return [
    { value: "", label: "— select checkpoint —" },
    ...checkpoints
      .filter((c) => c.type === "styletts2")
      .map((c) => ({ value: c.id, label: c.name })),
  ];
}
