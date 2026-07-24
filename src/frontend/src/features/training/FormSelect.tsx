import { useState } from "react";

import { Select, type Option } from "@/shared/ui/Select";

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

export function opts(items: (string | Option)[]): Option[] {
  return items.map((o) =>
    typeof o === "string" ? { value: o, label: o } : o,
  );
}
