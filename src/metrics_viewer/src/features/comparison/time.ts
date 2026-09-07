interface RelativeSeries {
  rel: number[] | null;
}

const TIME_UNITS = [
  { minimum: 86_400, divisor: 86_400, suffix: "d", title: "days" },
  { minimum: 3_600, divisor: 3_600, suffix: "h", title: "hours" },
  { minimum: 60, divisor: 60, suffix: "min", title: "minutes" },
  { minimum: 0, divisor: 1, suffix: "s", title: "seconds" },
];

/** Labels elapsed seconds in a unit suited to the visible duration without changing x values. */
export function relativeTimeTicks(series: RelativeSeries[]) {
  const values = series.flatMap((item) => item.rel ?? []);
  if (values.length === 0) return {};
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const unit = TIME_UNITS.find((candidate) => maximum - minimum >= candidate.minimum) ?? TIME_UNITS.at(-1)!;
  const step = niceStep(Math.max((maximum - minimum) / 5, unit.divisor / 10));
  const first = Math.ceil(minimum / step) * step;
  const tickvals = Array.from(
    { length: Math.max(1, Math.floor((maximum - first) / step) + 1) },
    (_, index) => first + index * step,
  );
  return {
    title: { text: `time (${unit.title})`, standoff: 8 },
    automargin: true,
    tickmode: "array" as const,
    tickvals,
    ticktext: tickvals.map((value) => formatRelativeTime(value, unit)),
  };
}

export function formatRelativeTime(value: number, selected = TIME_UNITS.find((unit) => value >= unit.minimum) ?? TIME_UNITS.at(-1)!): string {
  return `${(value / selected.divisor).toLocaleString(undefined, { maximumFractionDigits: 1 })} ${selected.suffix}`;
}

function niceStep(value: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return factor * magnitude;
}
