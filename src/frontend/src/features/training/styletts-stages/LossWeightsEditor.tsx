import { NumberInput } from "@/shared/ui/form/NumberInput";

import {
  TRAINING_LOSSES,
  type TrainingLossWeights,
} from "./templates";

export function LossWeightsEditor({
  weights,
  onChange,
}: {
  weights: TrainingLossWeights;
  onChange: (weights: TrainingLossWeights) => void;
}) {
  return (
    <div className="mt-4">
      <div className="mb-2 text-xs font-semibold text-txt-mute">
        Loss weights
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {TRAINING_LOSSES.map((loss) => (
          <label key={loss} className="flex flex-col gap-1">
            <span className="text-[11px] font-semibold text-txt">
              {loss.replaceAll("_", " ")}
            </span>
            <NumberInput
              value={weights[loss]}
              min={0}
              step={0.1}
              decimals={1}
              onChange={(weight) => onChange({
                ...weights,
                [loss]: weight,
              })}
            />
          </label>
        ))}
      </div>
    </div>
  );
}
