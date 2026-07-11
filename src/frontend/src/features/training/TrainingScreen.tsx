import { useEffect } from "react";

import { Tabs } from "@/shared/ui/Tabs";
import { useWorkflowSchemaQuery } from "../workflows/query";
import type { WorkflowGraph } from "../workflows/types";

import { FormSection } from "./FormSection";
import { createTrainingGraph, TRAINING_OPTIONS, TRAINING_WORKFLOWS } from "./logic";
import { QueueCard } from "./QueueCard";
import { MosModelForm } from "./MosModelForm";
import { SmallModelForm } from "./SmallModelForm";
import { StyleTtsForm } from "./StyleTtsForm";
import { useTraining } from "./store";
import type { TrainTab } from "./store";

export function TrainingScreen() {
  const trainTab = useTraining((s) => s.trainTab);
  const setTrainTab = useTraining((s) => s.setTrainTab);
  const graph = useTraining((s) => s.graphsByTab[trainTab]);
  const ensureGraph = useTraining((s) => s.ensureGraph);
  const setGraph = useTraining((s) => s.setGraph);
  const schemaQuery = useWorkflowSchemaQuery();
  const spec = TRAINING_WORKFLOWS[trainTab];

  useEffect(() => {
    if (!schemaQuery.data) return;
    ensureGraph(trainTab, createTrainingGraph(schemaQuery.data, spec));
  }, [schemaQuery.data, spec, trainTab, ensureGraph]);

  const updateGraph = (next: WorkflowGraph) => setGraph(trainTab, next);

  return (
    <div className="mx-auto max-w-[1180px] px-7 pt-6 pb-[120px]">
      <Tabs
        value={trainTab}
        onChange={(v) => setTrainTab(v as TrainTab)}
        options={TRAINING_OPTIONS}
        className="mb-[22px]"
      />

      <div className="grid grid-cols-[1fr_300px] items-start gap-6">
        <div className="flex min-w-0 flex-col gap-3.5">
          {schemaQuery.isLoading ? <FormSection title="Training workflow" tag="Schema">Loading node schema...</FormSection> : null}
          {schemaQuery.error ? <FormSection title="Training workflow" tag="Error">Could not load workflow schema.</FormSection> : null}
          {schemaQuery.data && graph ? (
            trainTab === "styletts" ? (
              <StyleTtsForm schema={schemaQuery.data} graph={graph} spec={spec} onChange={updateGraph} />
            ) : trainTab === "mos" ? (
              <MosModelForm schema={schemaQuery.data} graph={graph} spec={spec} onChange={updateGraph} />
            ) : (
              <SmallModelForm variant={trainTab} schema={schemaQuery.data} graph={graph} spec={spec} onChange={updateGraph} />
            )
          ) : null}
        </div>
        <div className="sticky top-0 flex flex-col gap-3.5">
          <QueueCard schema={schemaQuery.data ?? null} graph={graph ?? null} />
        </div>
      </div>
    </div>
  );
}
