import { createFileRoute } from "@tanstack/react-router";

import { App } from "@/app/App";
import { searchSchema } from "@/features/viewer/search";

export const Route = createFileRoute("/")({
  ssr: false,
  validateSearch: searchSchema,
  component: App,
});
