export type HistogramBars = {
  centers: number[];
  widths: number[];
  counts: number[];
  ranges: string[];
};

export function histogramBars(edges: number[], counts: number[], underflow: number, overflow: number): HistogramBars {
  const centers = counts.map((_, index) => (edges[index]! + edges[index + 1]!) / 2);
  const widths = counts.map((_, index) => edges[index + 1]! - edges[index]!);
  const ranges = counts.map((_, index) => `${fmt(edges[index]!)} – ${fmt(edges[index + 1]!)}`);
  const barCounts = [...counts];
  if (underflow > 0) {
    centers.unshift(edges[0]! - widths[0]! / 2);
    widths.unshift(widths[0]!);
    ranges.unshift(`< ${fmt(edges[0]!)}`);
    barCounts.unshift(underflow);
  }
  if (overflow > 0) {
    centers.push(edges.at(-1)! + widths.at(-1)! / 2);
    widths.push(widths.at(-1)!);
    ranges.push(`≥ ${fmt(edges.at(-1)!)}`);
    barCounts.push(overflow);
  }
  return { centers, widths, counts: barCounts, ranges };
}

function fmt(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && abs < 0.1) return value.toFixed(2);
  if (abs < 100) return value.toFixed(1);
  return Math.round(value).toLocaleString();
}
