import { useEffect } from "react";

import { EmptyState } from "@/shared/ui/EmptyState";
import { WorkflowCanvas } from "./components/WorkflowCanvas";
import { WorkflowInspector } from "./components/WorkflowInspector";
import { useRunsQuery, useWorkflowSchemaQuery } from "./query";
import { useWorkflowSocket } from "./socket";
import { useWorkflowStore } from "./store";
import { transcriptTemplate } from "./templates";

export function WorkflowsScreen() {
  const schemaQuery = useWorkflowSchemaQuery();
  const runsQuery = useRunsQuery();
  const { schema, graph, setSchema, setGraph, applyRunnerStatus } = useWorkflowStore();
  useWorkflowSocket();

  useEffect(() => {
    if (!schemaQuery.data || schema) return;
    setSchema(schemaQuery.data);
    setGraph(transcriptTemplate(schemaQuery.data));
  }, [schemaQuery.data, schema, setGraph, setSchema]);

  useEffect(() => {
    if (runsQuery.data) applyRunnerStatus(runsQuery.data.runs);
  }, [applyRunnerStatus, runsQuery.data]);

  if (schemaQuery.isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState icon="loader" title="Loading workflow schema" description="Fetching runner-owned node definitions." />
      </div>
    );
  }
  if (schemaQuery.error || !schema) {
    return (
      <div className="flex h-full items-center justify-center">
        <EmptyState icon="alert" title="Workflow schema unavailable" description="The backend did not return node metadata." />
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      <WorkflowCanvas />
      {graph.nodes.length ? <WorkflowInspector /> : null}
    </div>
  );
}
