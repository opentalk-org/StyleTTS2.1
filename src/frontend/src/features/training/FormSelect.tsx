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
