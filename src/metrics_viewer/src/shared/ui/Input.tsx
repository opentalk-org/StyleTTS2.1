import { Check, Search, X } from "lucide-react";
import { useLayoutEffect, useRef, type InputHTMLAttributes, type ReactNode, type TextareaHTMLAttributes } from "react";

import { cn } from "./cn";

export interface SearchInputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type"> {
  /** Accessible name; the icon alone is not one. */
  label: string;
  value: string;
  onValue: (value: string) => void;
}

export function SearchInput({ label, value, onValue, className, ...rest }: SearchInputProps) {
  return (
    <div
      className={cn(
        // focus-ring-owner: this wrapper draws the focus treatment, so the inner
        // input must not paint a second ring inside it.
        "focus-ring-owner group flex h-8 min-w-0 items-center gap-2 rounded-lg border border-line-hover",
        "bg-inset px-2.5 text-fg-muted transition-colors duration-150 ease-out",
        "focus-within:border-accent/65 focus-within:shadow-[0_0_0_3px_rgb(99_102_241/0.12)]",
        className,
      )}
    >
      <Search size={14} className="shrink-0 transition-colors group-focus-within:text-fg-secondary" />
      <input
        type="text"
        aria-label={label}
        value={value}
        onChange={(event) => onValue(event.target.value)}
        className="min-w-0 flex-1 bg-transparent text-sm text-fg placeholder:text-fg-muted focus:outline-none"
        {...rest}
      />
      {value.length === 0 ? null : (
        <button
          type="button"
          aria-label={`Clear ${label.toLowerCase()}`}
          title="Clear"
          onClick={() => onValue("")}
          className="grid size-4 shrink-0 place-items-center rounded-full text-fg-muted transition-colors duration-150 hover:bg-surface hover:text-fg"
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Grows with its content instead of scrolling inside a fixed box. */
  autoGrow?: boolean;
}

export function Textarea({ autoGrow = false, className, value, ...rest }: TextareaProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!autoGrow || node === null) return;
    // Collapse first: scrollHeight never shrinks below the current height.
    node.style.height = "auto";
    node.style.height = `${node.scrollHeight}px`;
  }, [autoGrow, value]);

  return (
    <textarea
      ref={ref}
      value={value}
      spellCheck={false}
      className={cn(
        "block w-full border-0 bg-transparent p-3 font-mono text-[13px] leading-relaxed",
        "text-fg-secondary caret-accent-bright focus:outline-none",
        autoGrow ? "resize-none overflow-hidden" : "resize-y",
        className,
      )}
      {...rest}
    />
  );
}

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  children: ReactNode;
}

export function Checkbox({ children, className, ...rest }: CheckboxProps) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-fg-secondary",
        "transition-colors duration-150 ease-out hover:bg-surface hover:text-fg",
        className,
      )}
    >
      <input type="checkbox" className="peer sr-only" {...rest} />
      <span
        aria-hidden
        className={cn(
          "grid size-4 shrink-0 place-items-center rounded-[4px] border border-line-hover bg-inset",
          "text-transparent transition-colors duration-150 ease-out",
          "peer-checked:border-accent peer-checked:bg-accent peer-checked:text-white",
          "peer-focus-visible:shadow-[0_0_0_3px_rgb(99_102_241/0.18)]",
        )}
      >
        <Check size={11} />
      </span>
      {children}
    </label>
  );
}

export interface FieldProps {
  label: ReactNode;
  /** Current value, shown in mono next to the label for range-style controls. */
  value?: ReactNode;
  /**
   * Set when the control is a composite (e.g. a segmented control) rather than a
   * single labelable element — clicking a <label> would otherwise activate its
   * first button. Composite controls carry their own accessible name.
   */
  group?: boolean;
  children: ReactNode;
  className?: string;
}

/** Label-left / control-right row used inside settings popovers. */
export function Field({ label, value, group = false, children, className }: FieldProps) {
  const Row = group ? "div" : "label";
  return (
    <Row className={cn("flex min-h-9 items-center justify-between gap-3 text-sm text-fg-secondary", className)}>
      <span className="flex shrink-0 items-baseline gap-1.5">
        {label}
        {value === undefined ? null : (
          <span className="font-mono text-xs tabular-nums text-fg">{value}</span>
        )}
      </span>
      {children}
    </Row>
  );
}

export interface RangeProps {
  min: number;
  max: number;
  step: number;
  value: number;
  onValue: (value: number) => void;
  className?: string;
  "aria-label"?: string;
}

export function Range({ min, max, step, value, onValue, className, ...rest }: RangeProps) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onValue(Number(event.target.value))}
      className={cn("w-[170px]", className)}
      {...rest}
    />
  );
}
