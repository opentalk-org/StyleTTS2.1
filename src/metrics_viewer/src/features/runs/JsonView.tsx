import { ChevronRight } from "lucide-react";
import { useState } from "react";

import { cn } from "@/shared/ui";
import type { JsonValue as JsonData } from "@/shared/types";

export function JsonView({ value }: { value: JsonData }) {
  return (
    <div className="p-4 font-mono text-xs leading-5 text-fg">
      <JsonValue value={value} depth={0} label={null} />
    </div>
  );
}

interface JsonValueProps {
  value: JsonData;
  depth: number;
  label: string | null;
}

function JsonValue({ value, depth, label }: JsonValueProps) {
  const structured = typeof value === "object" && value !== null;
  const [open, setOpen] = useState(depth < 2);
  if (!structured) return <ScalarRow label={label} value={value} depth={depth} />;

  const entries = Array.isArray(value)
    ? value.map((item, index) => [String(index), item] as const)
    : Object.entries(value);
  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1 rounded-sm py-0.5 text-left hover:bg-hover"
        style={{ paddingLeft: depth * 16 }}
      >
        <ChevronRight size={12} className={cn("shrink-0 text-fg-muted transition-transform", open ? "rotate-90" : "")} />
        {label === null ? null : <span className="text-accent">{label}</span>}
        <span className="text-fg-muted">{Array.isArray(value) ? `Array(${entries.length})` : `{${entries.length}}`}</span>
      </button>
      {open ? entries.map(([key, item]) => <JsonValue key={key} label={key} value={item} depth={depth + 1} />) : null}
    </div>
  );
}

function ScalarRow({ label, value, depth }: JsonValueProps) {
  return (
    <div className="flex min-h-6 items-start gap-2 rounded-sm py-0.5 hover:bg-hover" style={{ paddingLeft: depth * 16 + 17 }}>
      {label === null ? null : <span className="shrink-0 text-accent">{label}</span>}
      <span className={scalarColor(value)}>{formatScalar(value)}</span>
    </div>
  );
}

function formatScalar(value: unknown) {
  if (typeof value === "string") return JSON.stringify(value);
  if (value === null) return "null";
  return String(value);
}

function scalarColor(value: unknown) {
  if (typeof value === "string") return "break-all text-succeeded";
  if (typeof value === "number") return "text-queued";
  if (typeof value === "boolean") return "text-accent";
  return "text-fg-muted";
}
