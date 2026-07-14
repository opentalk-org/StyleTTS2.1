import type { WorkflowSchema } from "../workflows/types.ts";
import { catalogItemsFromSchema, groupCatalogItems } from "./catalog.ts";

const schema: WorkflowSchema = {
  types: {},
  nodes: {
    CatalogDownload: {
      type: "CatalogDownload",
      category: "assets",
      description: "",
      is_input: false,
      inputs: {},
      outputs: {},
      settings: {
        "x-catalog-items": [
          { name: "First", file: "first.bin", group: "One", catalog_key: "models", item: "first" },
          { name: "Second", file: "second.bin", group: "Two", catalog_key: "models", item: "second" },
        ],
      },
      settings_defaults: {},
      runtime: {},
      runtime_defaults: {},
    },
  },
  runtime_config: {},
  runtime_config_defaults: {},
};

const items = catalogItemsFromSchema(schema);
const groups = groupCatalogItems(items);

if (items.length !== 2) throw new Error(`Expected two catalog items, received ${items.length}`);
if (groups.One?.length !== 1 || groups.Two?.length !== 1) throw new Error("Expected one item in each dynamic group");
