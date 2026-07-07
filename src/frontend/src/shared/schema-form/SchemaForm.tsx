import { Fragment, type ReactNode } from "react";

import { SchemaField } from "./SchemaField";
import type { JsonSchema, SchemaValues } from "./types";

export type FieldOverride = (context: {
  name: string;
  label: string;
  description?: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) => ReactNode;

export function SchemaForm({
  schema,
  values,
  onChange,
  overrides,
}: {
  schema: JsonSchema;
  values: SchemaValues;
  onChange: (values: SchemaValues) => void;
  overrides?: Record<string, FieldOverride>;
}) {
  const properties = schema.properties ?? {};
  const set = (key: string, value: unknown) => onChange({ ...values, [key]: value });

  return (
    <div className="grid gap-3.5">
      {Object.entries(properties).map(([name, prop]) => {
        const override = overrides?.[name];
        if (override) {
          return (
            <Fragment key={name}>
              {override({ name, label: prop.title ?? name, description: prop.description, value: values[name], onChange: (value) => set(name, value) })}
            </Fragment>
          );
        }
        return <SchemaField key={name} name={name} schema={prop} root={schema} value={values[name]} onChange={(value) => set(name, value)} />;
      })}
    </div>
  );
}
