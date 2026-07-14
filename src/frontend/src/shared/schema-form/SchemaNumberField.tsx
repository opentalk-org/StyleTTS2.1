import { useEffect, useRef, useState } from "react";

import { Input } from "../ui/Input";
import { valueToText } from "./logic";
import { numericDraftValue, numericStep, type NumericSchemaType } from "./number";
import type { JsonSchema } from "./types";


export function SchemaNumberField({
  type,
  schema,
  value,
  onChange,
}: {
  type: NumericSchemaType;
  schema: JsonSchema;
  value: unknown;
  onChange: (value: number) => void;
}) {
  const [draft, setDraft] = useState(() => valueToText(value, schema));
  const editing = useRef(false);

  useEffect(() => {
    if (!editing.current) setDraft(valueToText(value, schema));
  }, [schema, value]);

  const updateDraft = (next: string) => {
    setDraft(next);
    const parsed = numericDraftValue(next, type);
    if (parsed !== undefined) onChange(parsed);
  };

  return (
    <Input
      filled
      className="h-9"
      type="number"
      value={draft}
      min={schema.minimum}
      max={schema.maximum}
      step={numericStep(type)}
      onFocus={() => { editing.current = true; }}
      onBlur={() => {
        editing.current = false;
        setDraft(valueToText(value, schema));
      }}
      onChange={(event) => updateDraft(event.target.value)}
    />
  );
}
