import type { WordAlignment } from "../api";

export type PlacedAlignment = {
  word: WordAlignment;
  row: number;
};

export function activeAlignment(words: WordAlignment[] | null | undefined, playPos: number): WordAlignment | null {
  if (!words) return null;
  return words.reduce<WordAlignment | null>((active, word) => {
    if (playPos < word.start || playPos >= word.end) return active;
    return active === null || word.start >= active.start ? word : active;
  }, null);
}

export function placeAlignmentRows(words: WordAlignment[]): PlacedAlignment[] {
  const rowEnds: number[] = [];
  return [...words]
    .sort((left, right) => left.start - right.start || left.end - right.end)
    .map((word) => {
      let row = rowEnds.findIndex((end) => word.start >= end);
      if (row === -1) {
        row = rowEnds.length;
        rowEnds.push(word.end);
      } else {
        rowEnds[row] = word.end;
      }
      return { word, row };
    });
}
