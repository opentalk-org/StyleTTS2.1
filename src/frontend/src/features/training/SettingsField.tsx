import { SchemaField } from "@/shared/schema-form/SchemaField";
import type { JsonSchema, SchemaValues } from "@/shared/schema-form/types";
import { Field } from "@/shared/ui/form/Field";
import { NumberInput } from "@/shared/ui/form/NumberInput";

type Props = {
  schema: JsonSchema;
  values: SchemaValues;
  name: string;
  onChange: (values: SchemaValues) => void;
};

export function SettingField({ schema, values, name, onChange }: Props) {
  const prop = settingProp(schema, name);
  return (
    <SchemaField
      name={name}
      schema={prop}
      root={schema}
      value={values[name]}
      onChange={(value) => onChange({ ...values, [name]: value })}
    />
  );
}

export function SettingNumberInput({
  schema,
  values,
  name,
  hint,
  step = 1,
  onChange,
}: Props & {
  hint?: string;
  step?: number;
}) {
  const prop = settingProp(schema, name);
  return (
    <Field label={settingTitle(prop, name)} hint={hint}>
      <NumberInput
        value={Number(values[name])}
        min={prop.minimum}
        max={prop.maximum}
        step={step}
        onChange={(value) => onChange({ ...values, [name]: value })}
      />
    </Field>
  );
}

export function settingTitle(prop: JsonSchema, name: string) {
  return prop.title ?? name;
}

export function settingLabel(schema: JsonSchema, name: string) {
  return settingTitle(settingProp(schema, name), name);
}

function settingProp(schema: JsonSchema, name: string): JsonSchema {
  const prop = schema.properties?.[name];
  if (!prop) throw new Error(`Training setting is not declared by node schema: ${name}`);
  return prop;
}
