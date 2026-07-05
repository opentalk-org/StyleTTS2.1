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
    <div className="rounded-md border border-line bg-panel-2 p-3">
      <div className="mb-2 text-[12px] font-bold uppercase tracking-wider text-txt-mute">{name}</div>
      <SchemaForm schema={{ ...schema, $defs: root.$defs }} values={value} onChange={onChange} />
    </div>
  );
}
