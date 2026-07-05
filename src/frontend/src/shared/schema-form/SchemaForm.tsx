import { SchemaField } from "./SchemaField";
import type { JsonSchema, SchemaValues } from "./types";

export function SchemaForm({
  schema,
  values,
  onChange,
}: {
  schema: JsonSchema;
  values: SchemaValues;
  onChange: (values: SchemaValues) => void;
}) {
  const properties = schema.properties ?? {};
  const set = (key: string, value: unknown) => onChange({ ...values, [key]: value });

  return (
    <div className="grid gap-3">
      {Object.entries(properties).map(([name, prop]) => (
        <SchemaField key={name} name={name} schema={prop} root={schema} value={values[name]} onChange={(value) => set(name, value)} />
      ))}
    </div>
  );
}
