import { EmptyState } from "@/shared/ui/EmptyState";

/**
 * Jobs surface. Intentionally an empty placeholder for now — the live job table
 * is part of a later build pass; only the nav entry is wired up.
 */
export function JobsScreen() {
  return (
    <div className="flex h-full items-center justify-center">
      <EmptyState
        icon="list-checks"
        title="Jobs"
        description="The job monitor is part of the next build pass."
      />
    </div>
  );
}
