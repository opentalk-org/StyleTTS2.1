import { Input } from "../ui/Input";
import { Select } from "../ui/Select";
import { Field } from "../ui/form/Field";
import { Toggle } from "../ui/form/Toggle";
import { resolveSchemaRef, schemaType, textToValue, valueToText } from "./logic";
import { SchemaMapField } from "./SchemaMapField";
import { SchemaObjectField } from "./SchemaObjectField";
import type { JsonSchema, SchemaValues } from "./types";

type Props = {
  name: string;
  schema: JsonSchema;
  root: JsonSchema;
  value: unknown;
  onChange: (value: unknown) => void;
};

export function SchemaField({ name, schema, root, value, onChange }: Props) {
  const resolved = resolveSchemaRef(schema, root);
  const type = schemaType(resolved);
  const label = schema.title ?? resolved.title ?? name;
  if (type === "object") {
    if (resolved.properties) {
      return <SchemaObjectField name={label} schema={resolved} root={root} value={(value ?? {}) as SchemaValues} onChange={onChange} />;
    }
    return <SchemaMapField name={label} schema={resolved} value={(value ?? {}) as SchemaValues} onChange={onChange} />;
  }
  if (type === "boolean") {
    return (
      <label className="flex cursor-pointer items-center gap-3 py-1">
        <span className="flex-1 text-[13px] font-semibold text-txt">{label}</span>
        <Toggle checked={Boolean(value)} onChange={onChange} />
      </label>
    );
  }
  if (resolved.enum) {
    return (
      <Field label={label}>
        <Select value={String(value ?? resolved.enum[0])} options={resolved.enum.map((item) => ({ value: item, label: item }))} onChange={onChange} />
      </Field>
    );
  }
  return (
    <Field label={label}>
      <Input
        filled
        className="h-9"
        type={type === "integer" || type === "number" ? "number" : "text"}
        value={valueToText(value, resolved)}
        onChange={(event) => onChange(textToValue(event.target.value, resolved))}
      />
    </Field>
  );
}
