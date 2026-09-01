import { useState } from "react";

import { cn } from "./cn";
import { Popover } from "./Popover";

export interface ColorPickerProps {
  /** Current color; also the swatch fill. */
  value: string;
  colors: readonly string[];
  onValue: (color: string) => void;
  /** Restores the palette default. Omitted when there is nothing to reset. */
  onReset?: () => void;
  /** Accessible name, e.g. "Plot color for aurora-004". */
  label: string;
  className?: string;
}

/** Small circular swatch that opens a compact palette. */
export function ColorPicker({ value, colors, onValue, onReset, label, className }: ColorPickerProps) {
  const [open, setOpen] = useState(false);

  return (
    <Popover
      open={open}
      onClose={() => setOpen(false)}
      align="start"
      portal
      className={cn("shrink-0", className)}
      panelClassName="w-max"
      trigger={
        <button
          type="button"
          title={label}
          aria-label={label}
          aria-expanded={open}
          onClick={(event) => {
            event.stopPropagation();
            setOpen(!open);
          }}
          className="group/swatch grid size-5 shrink-0 place-items-center rounded-full"
        >
          {/* Small dot, larger hit area: the target stays clickable at this density. */}
          <span
            className={cn(
              "block size-2.5 rounded-full ring-offset-2 ring-offset-elevated",
              "transition-[box-shadow] duration-150 ease-out",
              open ? "ring-2 ring-accent-bright" : "group-hover/swatch:ring-2 group-hover/swatch:ring-line-hover",
            )}
            style={{ background: value }}
          />
        </button>
      }
    >
      <div
        className="flex flex-col gap-1.5"
        onClick={(event) => event.stopPropagation()}
        role="presentation"
      >
        <div className="grid grid-cols-5 gap-1">
          {colors.map((color) => (
            <button
              key={color}
              type="button"
              title={color}
              aria-label={color}
              aria-pressed={color === value}
              onClick={() => {
                onValue(color);
                setOpen(false);
              }}
              className={cn(
                "size-5 rounded-full transition-transform duration-150 ease-out hover:scale-110",
                color === value ? "ring-2 ring-fg ring-offset-2 ring-offset-elevated" : "",
              )}
              style={{ background: color }}
            />
          ))}
        </div>
        {onReset === undefined ? null : (
          <button
            type="button"
            onClick={() => {
              onReset();
              setOpen(false);
            }}
            className="rounded-md px-1 py-1 text-center text-[11px] text-fg-muted transition-colors duration-150 hover:bg-surface hover:text-fg-secondary"
          >
            Reset to default
          </button>
        )}
      </div>
    </Popover>
  );
}
