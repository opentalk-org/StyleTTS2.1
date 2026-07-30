import { Button } from "@/shared/ui/Button";
import { NumberInput } from "@/shared/ui/form/NumberInput";
import { Toggle } from "@/shared/ui/form/Toggle";
import { Field } from "@/shared/ui/form/Field";
import { Select } from "@/shared/ui/Select";

import { LossWeightsEditor } from "./LossWeightsEditor";
import { ValidationSettingsEditor } from "./ValidationSettingsEditor";
import {
  cloneStage,
  STAGE_PRESETS,
  STAGE_TEMPLATES,
  TRAINABLE_MODULES,
  TRAINING_LOSSES,
  type TrainableModule,
  type TrainingLoss,
  type TrainingStageSpec,
} from "./templates";

export function StageScheduleEditor({
  stages,
  onChange,
}: {
  stages: TrainingStageSpec[];
  onChange: (stages: TrainingStageSpec[]) => void;
}) {
  const replace = (index: number, stage: TrainingStageSpec) => {
    onChange(stages.map((item, position) => (
      position === index ? stage : item
    )));
  };
  const move = (index: number, offset: number) => {
    const target = index + offset;
    if (target < 0 || target >= stages.length) return;
    const next = [...stages];
    [next[index], next[target]] = [next[target]!, next[index]!];
    onChange(next);
  };

  return (
    <div className="flex flex-col gap-4">
      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-txt-mute">Schedule presets</div>
        <div className="flex flex-wrap gap-2">
          {STAGE_PRESETS.map((preset) => (
            <Button
              key={preset.label}
              size="sm"
              onClick={() => onChange(preset.indexes.map(
                (index) => cloneStage(STAGE_TEMPLATES[index]!),
              ))}
            >
              {preset.label}
            </Button>
          ))}
        </div>
      </div>
      <div className="text-xs text-txt-mute">
        {stages.length} stages · {stages.reduce((sum, stage) => sum + stage.steps, 0).toLocaleString()} total steps
      </div>
      <details className="group rounded-lg border border-line-2 bg-panel-2">
        <summary className="cursor-pointer select-none px-4 py-3 text-sm font-semibold text-txt">
          Edit full stage specifications
        </summary>
        <div className="flex flex-col gap-4 border-t border-line p-4">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-txt-mute">Add preconfigured stage</div>
            <div className="flex flex-wrap gap-2">
              {STAGE_TEMPLATES.map((template) => (
                <Button
                  key={template.name}
                  size="sm"
                  variant="ghost"
                  onClick={() => onChange([...stages, cloneStage(template)])}
                >
                  {template.name}
                </Button>
              ))}
            </div>
          </div>
          {stages.map((stage, index) => (
            <StageEditor
              key={`${index}-${stage.name}`}
              stage={stage}
              index={index}
              count={stages.length}
              onChange={(next) => replace(index, next)}
              onMove={(offset) => move(index, offset)}
              onRemove={() => onChange(stages.filter((_, position) => position !== index))}
            />
          ))}
        </div>
      </details>
    </div>
  );
}

function StageEditor({
  stage,
  index,
  count,
  onChange,
  onMove,
  onRemove,
}: {
  stage: TrainingStageSpec;
  index: number;
  count: number;
  onChange: (stage: TrainingStageSpec) => void;
  onMove: (offset: number) => void;
  onRemove: () => void;
}) {
  return (
    <div className="rounded-lg border border-line-2 bg-panel-2 p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-line pb-3">
        <div>
          <div className="text-sm font-semibold text-txt">Stage {index + 1}</div>
          <div className="mt-0.5 text-xs text-txt-mute">
            {stage.steps.toLocaleString()} training steps
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={index === 0} onClick={() => onMove(-1)}>Up</Button>
          <Button size="sm" disabled={index === count - 1} onClick={() => onMove(1)}>Down</Button>
          <Button size="sm" variant="danger" disabled={count === 1} onClick={onRemove}>Remove</Button>
        </div>
      </div>
      <div className="mb-4 grid gap-3 sm:grid-cols-[minmax(0,1fr)_12rem]">
        <Field label="Stage name">
          <input
            value={stage.name}
            onChange={(event) => onChange({ ...stage, name: event.target.value })}
            className="h-10 w-full rounded-md border-2 border-transparent bg-panel px-3 text-[13.5px] font-semibold text-txt outline-none focus:border-blue-500"
          />
        </Field>
        <Field label="Steps">
          <NumberInput
            value={stage.steps}
            min={1}
            step={1000}
            onChange={(steps) => onChange({ ...stage, steps })}
          />
        </Field>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="Prosody source">
          <Select
            value={stage.prosody_source}
            options={[
              { value: "ground_truth", label: "Ground truth F0 / N" },
              { value: "predicted", label: "Predicted F0 / N" },
            ]}
            onChange={(prosody_source) => onChange({
              ...stage,
              prosody_source: prosody_source as TrainingStageSpec["prosody_source"],
            })}
          />
        </Field>
        <Field label="Reconstruction target">
          <Select
            value={stage.reconstruction_target}
            options={[
              { value: "real_audio", label: "Real recording" },
              { value: "teacher_reconstruction", label: "Teacher reconstruction" },
            ]}
            onChange={(reconstruction_target) => onChange({
              ...stage,
              reconstruction_target: reconstruction_target as TrainingStageSpec["reconstruction_target"],
            })}
          />
        </Field>
        <Field label="Train discriminators">
          <div className="flex h-10 items-center">
            <Toggle
              checked={stage.train_discriminators}
              onChange={(train_discriminators) => onChange({
                ...stage,
                train_discriminators,
                enabled_losses: toggleRequiredAdversarial(
                  stage.enabled_losses,
                  train_discriminators,
                ),
              })}
            />
          </div>
        </Field>
      </div>
      <ValidationSettingsEditor
        value={stage.validation}
        onChange={(validation) => onChange({ ...stage, validation })}
      />
      <ChoiceGrid
        title="Trainable modules"
        choices={TRAINABLE_MODULES}
        selected={stage.trainable_modules}
        onChange={(trainable_modules) => onChange({ ...stage, trainable_modules })}
      />
      <ChoiceGrid
        title="Enabled losses"
        choices={TRAINING_LOSSES}
        selected={stage.enabled_losses}
        onChange={(enabled_losses) => onChange({
          ...stage,
          enabled_losses,
          train_discriminators: enabled_losses.includes("adversarial"),
        })}
      />
      <LossWeightsEditor
        weights={stage.loss_weights}
        onChange={(loss_weights) => onChange({ ...stage, loss_weights })}
      />
    </div>
  );
}

function ChoiceGrid<T extends TrainableModule | TrainingLoss>({
  title,
  choices,
  selected,
  onChange,
}: {
  title: string;
  choices: readonly T[];
  selected: T[];
  onChange: (selected: T[]) => void;
}) {
  return (
    <div className="mt-4">
      <div className="mb-2 text-xs font-semibold text-txt-mute">{title}</div>
      <div className="flex flex-wrap gap-2">
        {choices.map((choice) => (
          <Button
            key={choice}
            size="sm"
            variant={selected.includes(choice) ? "primary" : "secondary"}
            onClick={() => onChange(
              selected.includes(choice)
                ? selected.filter((item) => item !== choice)
                : [...selected, choice],
            )}
          >
            {choice.replaceAll("_", " ")}
          </Button>
        ))}
      </div>
    </div>
  );
}

function toggleRequiredAdversarial(
  losses: TrainingLoss[],
  enabled: boolean,
): TrainingLoss[] {
  if (enabled && !losses.includes("adversarial")) {
    return [...losses, "adversarial"];
  }
  return enabled ? losses : losses.filter((loss) => loss !== "adversarial");
}
