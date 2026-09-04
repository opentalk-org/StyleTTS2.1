import { createFileRoute } from "@tanstack/react-router";

import { artifactResponse } from "@/features/artifacts/server";

export const Route = createFileRoute("/api/artifacts/content")({
  server: {
    handlers: {
      GET: ({ request }) => {
        const url = new URL(request.url);
        return artifactResponse(
          url.searchParams.get("run_id") ?? "",
          url.searchParams.get("path") ?? "",
        );
      },
    },
  },
});
