export type ClassValue = string | false | null | undefined;

/** Joins class names, dropping falsy branches. Consumer classes come last. */
export function cn(...values: ClassValue[]): string {
  return values.filter((value): value is string => typeof value === "string" && value.length > 0).join(" ");
}
