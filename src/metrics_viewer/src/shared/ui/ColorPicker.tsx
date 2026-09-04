import { useState } from "react";

import { useDebouncedCommit } from "@/shared/debounce";

import { cn } from "./cn";
import { Popover } from "./Popover";
import { Tooltip } from "./Tooltip";

export interface ColorPaletteProps {
  value: string;
  colors: readonly string[];
  /** A preset was chosen; callers usually close the popover here. */
  onValue: (color: string) => void;
  /** The free colour input changed; fires continuously while dragging, so keep the popover open. */
  onCustom: (color: string) => void;
  onReset?: () => void;
}

/** The native input fires on every pixel of a drag and each commit redraws every chart. */
const CUSTOM_COMMIT_DELAY = 120;

/** Free colour input plus preset swatches; the caller owns the popover. */
export function ColorPalette({ value, colors, onValue, onCustom, onReset }: ColorPaletteProps) {
  const [draft, setDraft] = useDebouncedCommit(toHex(value), onCustom, CUSTOM_COMMIT_DELAY);

  return (
    <div className="flex flex-col gap-2 p-1" onClick={(event) => event.stopPropagation()} role="presentation">
      <label className="flex h-7 cursor-pointer items-center gap-2 rounded-sm px-0.5 hover:bg-hover">
        <span className="relative size-5 shrink-0 overflow-hidden rounded-sm border border-strong" style={{ background: draft }}>
          <input
            type="color"
            aria-label="Custom colour"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            className="absolute inset-0 size-full cursor-pointer opacity-0"
          />
        </span>
        <span className="text-xs text-fg-secondary">Custom colour</span>
        <span className="ml-auto font-mono text-[11px] text-fg-muted">{draft}</span>
      </label>
      <div className="grid grid-cols-6 gap-1.5">
        {colors.map((color) => (
          <button
            key={color}
            type="button"
            title={color}
            aria-label={color}
            aria-pressed={color === value}
            onClick={() => onValue(color)}
            className={cn(
              "size-5 rounded-full",
              color === value
                ? "outline-2 outline-offset-2 outline-fg"
                : "hover:outline-2 hover:outline-offset-2 hover:outline-strong",
            )}
            style={{ background: color }}
          />
        ))}
      </div>
      {onReset === undefined ? null : (
        <button
          type="button"
          onClick={onReset}
          className="rounded-sm px-1 py-1 text-center text-xs text-fg-muted hover:bg-hover hover:text-fg"
        >
          Reset to default
        </button>
      )}
    </div>
  );
}

/** Palette and override colours are always #rrggbb; the native input wants it lowercase. */
function toHex(color: string): string {
  return color.toLowerCase();
}

export interface ColorPickerProps extends Omit<ColorPaletteProps, "onCustom"> {
  label: string;
  className?: string;
}

export function ColorPicker({ value, colors, onValue, onReset, label, className }: ColorPickerProps) {
  const [open, setOpen] = useState(false);

  return (
    <Popover
      open={open}
      onClose={() => setOpen(false)}
      align="start"
      className={cn("shrink-0", className)}
      width={168}
      trigger={
        <Tooltip content={label}>
          <button
            type="button"
            aria-label={label}
            aria-expanded={open}
            onClick={(event) => {
              event.stopPropagation();
              setOpen(!open);
            }}
            className="grid size-6 shrink-0 place-items-center rounded-sm hover:bg-hover"
          >
            <span
              className={cn("block size-3 rounded-full", open ? "outline-2 outline-offset-2 outline-accent" : "")}
              style={{ background: value }}
            />
          </button>
        </Tooltip>
      }
    >
      <ColorPalette
        value={value}
        colors={colors}
        onValue={(color) => {
          onValue(color);
          setOpen(false);
        }}
        onCustom={onValue}
        onReset={
          onReset === undefined
            ? undefined
            : () => {
                onReset();
                setOpen(false);
              }
        }
      />
    </Popover>
  );
}
