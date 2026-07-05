import type { JsonSchema, SchemaValues } from "./types";

export function resolveSchemaRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (!schema.$ref) return schema;
  const name = schema.$ref.replace("#/$defs/", "");
  const defs = root.$defs;
  if (!defs || !defs[name]) throw new Error(`Unknown schema ref: ${schema.$ref}`);
  return defs[name];
}

export function schemaType(schema: JsonSchema): string {
  if (schema.type) return schema.type;
  const variant = schema.anyOf?.find((item) => item.type !== "null");
  return variant?.type ?? "string";
}

export function valueToText(value: unknown, schema: JsonSchema): string {
  if (value === null || value === undefined) return "";
  if (schemaType(schema) === "array") return Array.isArray(value) ? value.join(", ") : String(value);
  return String(value);
}

export function textToValue(value: string, schema: JsonSchema): unknown {
  const type = schemaType(schema);
  const nullable = schema.anyOf?.some((item) => item.type === "null") ?? false;
  if (value === "" && nullable) return null;
  if (value === "" && !["string", "array"].includes(type)) return null;
  if (type === "integer") return Number.parseInt(value, 10);
  if (type === "number") return Number.parseFloat(value);
  if (type === "boolean") return value === "true";
  if (type === "array") return value.split(",").map((item) => item.trim()).filter(Boolean);
  return value;
}

export function defaultValue(schema: JsonSchema): unknown {
  if (schema.default !== undefined) return schema.default;
  const type = schemaType(schema);
  if (type === "integer" || type === "number") return 0;
  if (type === "boolean") return false;
  if (type === "array") return [];
  if (type === "object") return {};
  return "";
}

export function nextMapKey(values: SchemaValues): string {
  let index = 1;
  while (`entry_${index}` in values) index += 1;
  return `entry_${index}`;
}
