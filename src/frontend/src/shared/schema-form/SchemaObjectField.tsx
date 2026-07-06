import { SchemaForm } from "./SchemaForm";
import type { JsonSchema, SchemaValues } from "./types";

export function SchemaObjectField({
  name,
  schema,
  root,
  value,
  onChange,
}: {
  name: string;
  schema: JsonSchema;
  root: JsonSchema;
  value: SchemaValues;
  onChange: (value: unknown) => void;
}) {
  return (
    <div className="grid gap-3.5 border-t border-line pt-4 first:border-t-0 first:pt-0">
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-[15px] font-bold tracking-tight text-txt">{name}</div>
        <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-blue-500">
          Group
        </div>
      </div>
      <SchemaForm schema={{ ...schema, $defs: root.$defs }} values={value} onChange={onChange} />
    </div>
  );
}
