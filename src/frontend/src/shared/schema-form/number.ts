export type NumericSchemaType = "integer" | "number";

export function numericDraftValue(draft: string, type: NumericSchemaType): number | undefined {
  if (!draft || draft.endsWith(".") || draft === "-" || draft === "+") return undefined;
  if (type === "integer" && !/^[+-]?\d+$/.test(draft)) return undefined;
  const value = Number(draft);
  return Number.isFinite(value) ? value : undefined;
}

export function numericStep(type: NumericSchemaType): 1 | "any" {
  return type === "integer" ? 1 : "any";
}
