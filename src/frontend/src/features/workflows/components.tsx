import { EmptyState } from "../../shared/ui/EmptyState";

/**
 * Node-graph workflow editor. Intentionally left as a placeholder for a later
 * build pass — the surface exists in nav but is not implemented yet.
 */
export function WorkflowsScreen() {
  return (
    <div className="flex h-full items-center justify-center">
      <EmptyState
        icon="workflow"
        title="Workflows"
        description="The node-graph editor is part of the next build pass."
      />
    </div>
  );
}
