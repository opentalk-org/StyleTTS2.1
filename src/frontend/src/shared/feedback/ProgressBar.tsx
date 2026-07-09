import { cn } from "../ui/cn";

type ProgressTone = "blue" | "emerald" | "red";

const FILL: Record<ProgressTone, string> = {
  blue: "bg-blue-500",
  emerald: "bg-emerald-500",
  red: "bg-red-500",
};

export function ProgressBar({
  value,
  tone = "blue",
  className,
}: {
  value: number;
  tone?: ProgressTone;
  className?: string;
}) {
  return (
    <div className={cn("h-1.5 overflow-hidden rounded-full bg-panel-2", className)}>
      <div
        className={cn("h-full rounded-full transition-[width] duration-300", FILL[tone])}
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}
