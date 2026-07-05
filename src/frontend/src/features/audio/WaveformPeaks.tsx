import { cn } from "@/shared/ui/cn";

export function WaveformPeaks({
  peaks,
  height,
  className,
}: {
  peaks: [number, number][];
  height: number;
  className?: string;
}) {
  if (!peaks.length) return null;
  return (
    <div className={cn("flex h-full w-full items-center overflow-hidden", className)} style={{ height }}>
      {peaks.map(([minimum, maximum], index) => {
        const top = ((1 - clampPeak(maximum)) / 2) * height;
        const bottom = ((1 - clampPeak(minimum)) / 2) * height;
        return (
          <span
            key={index}
            className="flex-1 bg-current"
            style={{
              height: Math.max(1, bottom - top),
              marginTop: top,
              marginBottom: Math.max(0, height - bottom),
            }}
          />
        );
      })}
    </div>
  );
}

function clampPeak(value: number): number {
  return Math.max(-1, Math.min(1, value));
}
