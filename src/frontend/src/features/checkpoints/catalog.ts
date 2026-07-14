import type { WorkflowSchema } from "../workflows/types";

export type CatalogItem = {
  name: string;
  file: string;
  group: string;
  catalogKey: string;
  item: string;
};

type SchemaCatalogItem = {
  name: string;
  file: string;
  group: string;
  catalog_key: string;
  item: string;
};

const CATALOG_ITEM_KEYS: (keyof SchemaCatalogItem)[] = ["name", "file", "group", "catalog_key", "item"];

function isSchemaCatalogItem(value: unknown): value is SchemaCatalogItem {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const entry = value as Record<string, unknown>;
  return CATALOG_ITEM_KEYS.every((key) => typeof entry[key] === "string");
}

export function catalogItemsFromSchema(schema: WorkflowSchema): CatalogItem[] {
  const rawItems = schema.nodes.CatalogDownload?.settings["x-catalog-items"];
  if (!Array.isArray(rawItems) || !rawItems.every(isSchemaCatalogItem)) {
    throw new Error("CatalogDownload x-catalog-items must be an array of catalog entries with string name, file, group, catalog_key, and item values");
  }
  return rawItems.map((entry) => ({
    name: entry.name,
    file: entry.file,
    group: entry.group,
    catalogKey: entry.catalog_key,
    item: entry.item,
  }));
}

export function groupCatalogItems(items: CatalogItem[]): Record<string, CatalogItem[]> {
  const groups: Record<string, CatalogItem[]> = {};
  for (const item of items) (groups[item.group] ??= []).push(item);
  return groups;
}
