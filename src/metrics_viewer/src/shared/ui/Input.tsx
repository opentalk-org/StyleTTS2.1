import { Check, Minus, Search, X } from "lucide-react";
import {
  useLayoutEffect,
  useRef,
  type InputHTMLAttributes,
  type ReactNode,
  type TextareaHTMLAttributes,
} from "react";

import { cn } from "./cn";
import { Kbd, Numeric } from "./Surface";

export interface SearchInputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type" | "size"> {
  label: string;
  value: string;
  onValue: (value: string) => void;
  shortcut?: string;
  size?: "md" | "lg";
}

export function SearchInput({ label, value, onValue, shortcut, size = "md", className, ...rest }: SearchInputProps) {
  return (
    <div
      className={cn(
        "focus-ring-owner group flex min-w-0 items-center gap-2 rounded-md border border-strong bg-inset px-2 text-fg-muted",
        "transition-colors duration-100 ease-out focus-within:border-accent",
        size === "lg" ? "h-control-lg" : "h-control",
        className,
      )}
    >
      <Search size={13} className="shrink-0" />
      <input
        type="text"
        aria-label={label}
        value={value}
        onChange={(event) => onValue(event.target.value)}
        className="min-w-0 flex-1 bg-transparent text-[13px] text-fg placeholder:text-fg-muted focus:outline-none"
        {...rest}
      />
      {value.length > 0 ? (
        <button
          type="button"
          aria-label={`Clear ${label.toLowerCase()}`}
          onClick={() => onValue("")}
          className="grid size-4 shrink-0 place-items-center rounded-sm text-fg-muted hover:bg-hover hover:text-fg"
        >
          <X size={12} />
        </button>
      ) : shortcut === undefined ? null : (
        <span className="hidden group-focus-within:hidden sm:inline">
          <Kbd>{shortcut}</Kbd>
        </span>
      )}
    </div>
  );
}

export interface TextInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "value"> {
  value: string;
  onValue: (value: string) => void;
}

export function TextInput({ value, onValue, className, ...rest }: TextInputProps) {
  return (
    <input
      type="text"
      value={value}
      onChange={(event) => onValue(event.target.value)}
      className={cn(
        "h-control min-w-0 rounded-md border border-strong bg-inset px-2 text-[13px] text-fg placeholder:text-fg-muted",
        "focus:border-accent focus:outline-none",
        className,
      )}
      {...rest}
    />
  );
}

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  autoGrow?: boolean;
}

export function Textarea({ autoGrow = false, className, value, ...rest }: TextareaProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!autoGrow || node === null) return;
    node.style.height = "auto";
    node.style.height = `${node.scrollHeight}px`;
  }, [autoGrow, value]);

  return (
    <textarea
      ref={ref}
      value={value}
      spellCheck={false}
      className={cn(
        "block w-full rounded-md border border-strong bg-inset p-3 font-mono text-[12.5px] leading-relaxed text-fg",
        "focus:border-accent focus:outline-none",
        autoGrow ? "resize-none overflow-hidden" : "resize-y",
        className,
      )}
      {...rest}
    />
  );
}

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  children?: ReactNode;
  indeterminate?: boolean;
}

export function Checkbox({ children, indeterminate = false, className, ...rest }: CheckboxProps) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 rounded-sm text-[13px] text-fg-secondary",
        children === undefined ? "" : "px-1 py-1 hover:bg-hover hover:text-fg",
        className,
      )}
    >
      <input type="checkbox" className="peer sr-only" {...rest} />
      <span
        aria-hidden
        className={cn(
          "grid size-3.5 shrink-0 place-items-center rounded-sm border border-strong bg-inset",
          "text-transparent transition-colors duration-100 ease-out",
          "peer-checked:border-accent peer-checked:bg-accent peer-checked:text-accent-fg",
          "peer-focus-visible:outline-2 peer-focus-visible:outline-accent peer-focus-visible:outline-offset-1",
          indeterminate ? "border-accent bg-accent text-accent-fg" : "",
        )}
      >
        {indeterminate ? <Minus size={10} strokeWidth={3} /> : <Check size={10} strokeWidth={3} />}
      </span>
      {children}
    </label>
  );
}

export interface FieldProps {
  label: ReactNode;
  value?: ReactNode;
  group?: boolean;
  children: ReactNode;
  className?: string;
}

/** Label + control on one row; `group` renders a div for controls that are not a single input. */
export function Field({ label, value, group = false, children, className }: FieldProps) {
  const Row = group ? "div" : "label";
  return (
    <Row className={cn("flex min-h-8 items-center justify-between gap-3 text-[13px] text-fg-secondary", className)}>
      <span className="flex shrink-0 items-baseline gap-1.5">
        {label}
        {value === undefined ? null : <Numeric className="text-fg">{value}</Numeric>}
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
  disabled?: boolean;
  className?: string;
  "aria-label"?: string;
}

export function Range({ min, max, step, value, onValue, disabled, className, ...rest }: RangeProps) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(event) => onValue(Number(event.target.value))}
      className={cn("w-40", className)}
      {...rest}
    />
  );
}
