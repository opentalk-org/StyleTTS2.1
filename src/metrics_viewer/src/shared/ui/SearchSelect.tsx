import { Check, ChevronsUpDown, Search } from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";

import { cn } from "./cn";
import { GroupLabel } from "./Surface";
import { Popover } from "./Popover";

export interface SearchOption {
  value: string;
  label: string;

  group?: string;

  hint?: string;
}

export interface SearchOptionListProps {
  options: SearchOption[];

  selected: string[];
  onSelect: (value: string) => void;

  multiple?: boolean;
  placeholder?: string;
  emptyMessage?: string;

  searchable?: boolean;
}





export function SearchOptionList({
  options,
  selected,
  onSelect,
  multiple = false,
  placeholder = "Search…",
  emptyMessage = "No match",
  searchable = true,
}: SearchOptionListProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const showSearch = searchable;

  const matches = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (normalized.length === 0) return options;
    return options.filter((option) =>
      `${option.label} ${option.value} ${option.hint ?? ""} ${option.group ?? ""}`
        .toLowerCase()
        .includes(normalized),
    );
  }, [options, query]);


  const groups = useMemo(() => {
    const byGroup = new Map<string, SearchOption[]>();
    for (const option of matches) {
      const key = option.group ?? "";
      byGroup.set(key, [...(byGroup.get(key) ?? []), option]);
    }
    return [...byGroup.entries()];
  }, [matches]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query]);


  useEffect(() => {
    if (!showSearch) rootRef.current?.focus();
  }, [showSearch]);

  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  function onKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (matches.length === 0) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((index) => (index + 1) % matches.length);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((index) => (index - 1 + matches.length) % matches.length);
    } else if (event.key === "Home") {
      event.preventDefault();
      setActiveIndex(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setActiveIndex(matches.length - 1);
    } else if (event.key === "Enter") {
      event.preventDefault();
      onSelect(matches[activeIndex].value);
      if (!multiple) setQuery("");
    }
  }

  let flatIndex = -1;

  return (
    <div
      ref={rootRef}
      className="flex w-full min-w-0 flex-col focus:outline-none"

      tabIndex={showSearch ? undefined : -1}
      onKeyDown={showSearch ? undefined : onKeyDown}
    >
      {showSearch ? (
        <div className="flex h-9 flex-none items-center gap-2 border-b border-line px-2.5 text-fg-muted">
          <Search size={13} className="shrink-0" />
          <input
            autoFocus
            type="text"
            role="combobox"
            aria-expanded
            aria-controls="search-option-list"
            aria-activedescendant={matches.length === 0 ? undefined : `option-${activeIndex}`}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            className="min-w-0 flex-1 bg-transparent text-sm text-fg placeholder:text-fg-muted focus:outline-none"
          />
          {query.length === 0 ? null : (
            <span className="shrink-0 font-mono text-[10px] tabular-nums text-fg-muted">
              {matches.length}
            </span>
          )}
        </div>
      ) : null}

      <div
        ref={listRef}
        id="search-option-list"
        role="listbox"
        aria-multiselectable={multiple}
        className="max-h-[min(420px,60vh)] min-h-0 overflow-auto p-1"
      >
        {matches.length === 0 ? (
          <p className="m-0 px-2 py-6 text-center text-xs text-fg-muted">{emptyMessage}</p>
        ) : null}
        {groups.map(([group, groupOptions]) => (
          <div key={group}>
            {group === "" ? null : (
              <GroupLabel className="block px-2 pt-2 pb-1">{group}</GroupLabel>
            )}
            {groupOptions.map((option) => {
              flatIndex += 1;
              const index = flatIndex;
              const isSelected = selected.includes(option.value);
              return (
                <button
                  key={option.value}
                  id={`option-${index}`}
                  data-index={index}
                  type="button"
                  role="option"
                  aria-selected={isSelected}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => onSelect(option.value)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
                    "transition-colors duration-150 ease-out",
                    index === activeIndex ? "bg-surface text-fg" : "text-fg-secondary",
                    isSelected ? "text-fg" : "",
                  )}
                >
                  {multiple ? (
                    <span
                      aria-hidden
                      className={cn(
                        "grid size-4 shrink-0 place-items-center rounded-[4px] border transition-colors duration-150",
                        isSelected
                          ? "border-accent bg-accent text-white"
                          : "border-line-hover bg-inset text-transparent",
                      )}
                    >
                      <Check size={11} />
                    </span>
                  ) : (
                    <Check
                      size={13}
                      aria-hidden
                      className={cn("shrink-0", isSelected ? "text-accent-bright" : "text-transparent")}
                    />
                  )}
                  <span className="min-w-0 flex-1 truncate">
                    <Highlight text={option.label} query={query} />
                  </span>
                  {option.hint === undefined ? null : (
                    <span className="shrink-0 font-mono text-[10px] text-fg-muted">{option.hint}</span>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}


function Highlight({ text, query }: { text: string; query: string }): ReactNode {
  const normalized = query.trim().toLowerCase();
  const start = normalized.length === 0 ? -1 : text.toLowerCase().indexOf(normalized);
  if (start === -1) return text;
  return (
    <>
      {text.slice(0, start)}
      <mark className="bg-transparent text-accent-bright">
        {text.slice(start, start + normalized.length)}
      </mark>
      {text.slice(start + normalized.length)}
    </>
  );
}

export interface SearchSelectProps {
  label: string;
  options: SearchOption[];
  value: string;
  onValue: (value: string) => void;
  placeholder?: string;
  emptyMessage?: string;
  searchable?: boolean;
  className?: string;
  align?: "start" | "end";
  portal?: boolean;
}


export function SearchSelect({
  label,
  options,
  value,
  onValue,
  placeholder,
  emptyMessage,
  searchable,
  className,
  align = "end",
  portal = false,
}: SearchSelectProps) {
  const [open, setOpen] = useState(false);
  const selected = options.find((option) => option.value === value);

  return (
    <Popover
      open={open}
      onClose={() => setOpen(false)}
      align={align}
      portal={portal}
      className={className}
      panelClassName="w-[min(360px,90vw)] overflow-hidden p-0"
      trigger={
        <button
          type="button"
          aria-label={label}
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
          className={cn(
            "flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-lg border bg-inset px-2.5",
            "text-sm transition-colors duration-150 ease-out",
            open
              ? "border-accent/65 text-fg shadow-[0_0_0_3px_rgb(99_102_241/0.12)]"
              : "border-line-hover text-fg hover:border-line-hover hover:bg-surface-hover",
          )}
        >
          <span className="min-w-0 truncate">{selected?.label ?? value}</span>
          <ChevronsUpDown size={13} className="shrink-0 text-fg-muted" />
        </button>
      }
    >
      <SearchOptionList
        options={options}
        selected={[value]}
        searchable={searchable}
        placeholder={placeholder}
        emptyMessage={emptyMessage}
        onSelect={(next) => {
          onValue(next);
          setOpen(false);
        }}
      />
    </Popover>
  );
}
