import { createFileRoute } from "@tanstack/react-router";

import { App } from "@/app/App";

export const Route = createFileRoute("/")({
  ssr: false,
  component: App,
});
