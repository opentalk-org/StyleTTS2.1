import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { defaultValue, nextMapKey, schemaType, textToValue, valueToText } from "./logic";
import type { JsonSchema, SchemaValues } from "./types";

export function SchemaMapField({
  name,
  schema,
  value,
  onChange,
}: {
  name: string;
  schema: JsonSchema;
  value: SchemaValues;
  onChange: (value: unknown) => void;
}) {
  const itemSchema = typeof schema.additionalProperties === "object" ? schema.additionalProperties : {};
  const setKey = (key: string, next: unknown) => onChange({ ...value, [key]: next });
  const rename = (oldKey: string, nextKey: string) => {
    if (!nextKey || oldKey === nextKey) return;
    const next = { ...value, [nextKey]: value[oldKey] };
    delete next[oldKey];
    onChange(next);
  };
  const remove = (key: string) => {
    const next = { ...value };
    delete next[key];
    onChange(next);
  };

  return (
    <div className="rounded-md border border-line bg-panel-2 p-3">
      <div className="mb-2 text-[12px] font-bold uppercase tracking-wider text-txt-mute">{name}</div>
      <div className="grid gap-2">
        {Object.entries(value).map(([key, item]) => (
          <div key={key} className="grid grid-cols-[1fr_1fr_auto] gap-2">
            <Input filled className="h-9" value={key} onChange={(event) => rename(key, event.target.value)} />
            <Input
              filled
              className="h-9"
              type={schemaType(itemSchema) === "string" ? "text" : "number"}
              value={valueToText(item, itemSchema)}
              onChange={(event) => setKey(key, textToValue(event.target.value, itemSchema))}
            />
            <Button variant="secondary" onClick={() => remove(key)}>Remove</Button>
          </div>
        ))}
        <Button variant="secondary" icon="plus" onClick={() => setKey(nextMapKey(value), defaultValue(itemSchema))}>
          Add entry
        </Button>
      </div>
    </div>
  );
}
