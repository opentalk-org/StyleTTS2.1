import { z } from "zod";

// Kept free of store imports: the route module evaluates on the server, the store touches localStorage.
export const TABS = ["charts", "compare", "media", "lineage", "graph"] as const;

export const searchSchema = z.object({
  project: z.string().optional(),
  runs: z.string().optional(),
  tab: z.enum(TABS).optional(),
  view: z.string().optional(),
});

export type ViewerSearch = z.infer<typeof searchSchema>;
