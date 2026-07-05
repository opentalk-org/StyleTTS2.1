import { cn } from "../ui/cn";

/** Deterministic pseudo-waveform heights (0–1) from a seed. */
export function waveBars(seed: number, n: number): number[] {
  const a: number[] = [];
  for (let i = 0; i < n; i++) {
    const noise = Math.abs(Math.sin(seed * 100 + i) % 1);
    const v = Math.abs(
      Math.sin(i * 0.21 + seed) * 0.6 + Math.sin(i * 0.07 + seed * 2) * 0.4 + noise * 0.25,
    );
    a.push(Math.min(1, 0.12 + v * 0.78));
  }
  return a;
}

/**
 * Bar-style waveform. Bars up to `progress` (0–1) render in the active color.
 */
export function WaveformBars({
  seed,
  bars = 48,
  progress = 0,
  height = 34,
  className,
}: {
  seed: number;
  bars?: number;
  progress?: number;
  height?: number;
  className?: string;
}) {
  const data = waveBars(seed, bars);
  return (
    <div className={cn("flex items-center gap-px", className)} style={{ height }}>
      {data.map((b, i) => (
        <div
          key={i}
          className={cn(
            "flex-1 rounded-[1px]",
            i / data.length <= progress ? "bg-blue-500" : "bg-gray-300",
          )}
          style={{ height: `${b * 100}%`, minHeight: 2 }}
        />
      ))}
    </div>
  );
}
