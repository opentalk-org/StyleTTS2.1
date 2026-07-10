import { useRef } from "react";

import { cn } from "@/shared/ui/cn";

export function formatAudioScore(score: number | null): string {
  return score === null ? "" : score.toFixed(3);
}

export function parseAudioScore(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const score = Number(trimmed);
  return Number.isFinite(score) ? score : null;
}

export function AudioScoreInput({
  value,
  disabled = false,
  label = "Score",
  className,
  onChange,
  onCommit,
  onCancel,
}: {
  value: string;
  disabled?: boolean;
  label?: string;
  className?: string;
  onChange: (value: string) => void;
  onCommit?: () => void | Promise<void>;
  onCancel: () => void;
}) {
  const skipBlur = useRef(false);

  return (
    <label className={cn("flex h-8 items-center rounded-md border border-line-2 bg-bg pl-2.5 pr-1 focus-within:border-blue-400", className)}>
      <span className="text-[10px] font-bold uppercase tracking-wide text-txt-mute">{label}</span>
      <input
        value={value}
        disabled={disabled}
        type="number"
        step="0.01"
        placeholder="—"
        title="Audio score"
        onChange={(event) => onChange(event.target.value)}
        onBlur={() => {
          if (skipBlur.current) {
            skipBlur.current = false;
            return;
          }
          void onCommit?.();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            event.currentTarget.blur();
          } else if (event.key === "Escape") {
            event.preventDefault();
            skipBlur.current = true;
            onCancel();
            event.currentTarget.blur();
          }
        }}
        className="h-6 w-24 min-w-0 rounded bg-transparent px-1 text-right font-mono text-[13px] font-semibold tabular-nums text-txt outline-none placeholder:text-txt-mute disabled:opacity-60"
      />
    </label>
  );
}
