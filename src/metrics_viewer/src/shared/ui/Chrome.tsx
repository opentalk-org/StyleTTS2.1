import { ChevronRight, X } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "./cn";
import { IconButton } from "./Button";
import { Tooltip } from "./Tooltip";

export interface ToolbarProps {
  start?: ReactNode;
  end?: ReactNode;
  sticky?: boolean;
  className?: string;
}

/** 40 px row with start/end slots; the standard header of every pane. */
export function Toolbar({ start, end, sticky = false, className }: ToolbarProps) {
  return (
    <div
      className={cn(
        "flex h-toolbar flex-none items-center justify-between gap-2 border-b border-line bg-surface px-3",
        sticky ? "sticky top-0 z-20" : "",
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-2">{start}</div>
      {end === undefined ? null : <div className="flex shrink-0 items-center gap-2">{end}</div>}
    </div>
  );
}

export interface TabItem<T extends string> {
  id: T;
  label: string;
  icon?: ReactNode;
  count?: number;
  disabledReason?: string;
}

export interface TabsProps<T extends string> {
  label: string;
  items: TabItem<T>[];
  value: T;
  onValue: (value: T) => void;
  end?: ReactNode;
}

export function Tabs<T extends string>({ label, items, value, onValue, end }: TabsProps<T>) {
  return (
    <nav
      aria-label={label}
      className="flex h-9 flex-none items-stretch justify-between gap-2 border-b border-line bg-surface px-3"
    >
      <div className="flex items-stretch gap-1">
        {items.map((item) => {
          const active = item.id === value;
          const disabled = item.disabledReason !== undefined;
          const button = (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={active}
              disabled={disabled}
              onClick={() => onValue(item.id)}
              className={cn(
                "relative flex items-center gap-1.5 px-2 text-[13px] font-medium",
                "transition-colors duration-100 ease-out",
                active ? "text-fg" : "text-fg-muted hover:text-fg-secondary",
                disabled ? "text-fg-disabled hover:text-fg-disabled" : "",
                "after:absolute after:inset-x-1 after:bottom-0 after:h-0.5 after:rounded-t-sm",
                active ? "after:bg-accent" : "after:bg-transparent",
              )}
            >
              {item.icon === undefined ? null : <span className="[&>svg]:size-3.5">{item.icon}</span>}
              {item.label}
              {item.count === undefined ? null : (
                <span className="rounded-sm bg-hover px-1 font-mono text-[11px] text-fg-muted">{item.count}</span>
              )}
            </button>
          );
          return disabled ? (
            <Tooltip key={item.id} content={item.disabledReason}>
              {button}
            </Tooltip>
          ) : (
            button
          );
        })}
      </div>
      {end === undefined ? null : <div className="flex items-center gap-2">{end}</div>}
    </nav>
  );
}

export interface SheetProps {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  actions?: ReactNode;
  width?: number;
  children: ReactNode;
}

/** Right-docked panel that pushes pane content instead of covering it. */
export function Sheet({ open, onClose, title, actions, width = 440, children }: SheetProps) {
  if (!open) return null;
  return (
    <aside
      aria-label={typeof title === "string" ? title : undefined}
      style={{ width }}
      className="flex min-h-0 flex-none flex-col border-l border-line bg-surface"
    >
      <header className="flex h-toolbar flex-none items-center justify-between gap-2 border-b border-line pr-2 pl-3">
        <h3 className="truncate text-sm font-semibold text-fg">{title}</h3>
        <div className="flex shrink-0 items-center gap-1">
          {actions}
          <IconButton label="Close" shortcut="Esc" size="sm" onClick={onClose}>
            <X size={13} />
          </IconButton>
        </div>
      </header>
      <div className="min-h-0 flex-1 overflow-auto">{children}</div>
    </aside>
  );
}

export interface CollapsibleProps {
  open: boolean;
  onToggle: () => void;
  title: ReactNode;
  count?: number;
  actions?: ReactNode;
  children: ReactNode;
}

/** Section with a chevron header; the body animates open and closed via grid rows. */
export function Collapsible({ open, onToggle, title, count, actions, children }: CollapsibleProps) {
  return (
    <section className="flex flex-col">
      <div className="group flex h-8 items-center gap-2 border-b border-line">
        <button
          type="button"
          aria-expanded={open}
          onClick={onToggle}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left text-[13px] font-semibold text-fg"
        >
          <ChevronRight
            size={14}
            className={cn("shrink-0 text-fg-muted transition-transform duration-200 ease-out", open ? "rotate-90" : "")}
          />
          <span className="truncate">{title}</span>
          {count === undefined ? null : (
            <span className="rounded-sm bg-hover px-1 font-mono text-[11px] font-normal text-fg-muted">{count}</span>
          )}
        </button>
        {actions}
      </div>
      <div
        className={cn("grid transition-[grid-template-rows] duration-200 ease-out", open ? "grid-rows-[1fr]" : "grid-rows-[0fr]")}
        aria-hidden={!open}
      >
        <div className={cn("min-h-0 overflow-hidden", open ? "pt-3" : "")}>{children}</div>
      </div>
    </section>
  );
}
