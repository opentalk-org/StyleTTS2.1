import { useState } from "react";

import { Select, type Option } from "@/shared/ui/Select";

/**
 * Local-state select for form fields whose value is a UI draft detail we do not
 * persist. Seeds from `defaultValue` and tracks its own selection.
 */
export function FormSelect({
  defaultValue,
  value,
  onChange,
  options,
}: {
  defaultValue: string;
  value?: string;
  onChange?: (value: string) => void;
  options: Option[];
}) {
  const [localValue, setLocalValue] = useState(defaultValue);
  return <Select value={value ?? localValue} onChange={onChange ?? setLocalValue} options={options} />;
}

/** Build select options from a mix of plain strings and explicit {value,label}. */
export function opts(items: (string | Option)[]): Option[] {
  return items.map((o) =>
    typeof o === "string" ? { value: o, label: o } : o,
  );
}
