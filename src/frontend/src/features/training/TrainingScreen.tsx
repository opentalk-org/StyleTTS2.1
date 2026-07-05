import { Tabs } from "@/shared/ui/Tabs";

import { QueueCard } from "./QueueCard";
import { SmallModelForm } from "./SmallModelForm";
import { StyleTtsForm } from "./StyleTtsForm";
import { useTraining } from "./store";
import type { TrainTab } from "./store";

const TABS = [
  { value: "styletts", label: "StyleTTS finetune" },
  { value: "f0", label: "F0 model" },
  { value: "asr", label: "ASR model" },
];

export function TrainingScreen() {
  const trainTab = useTraining((s) => s.trainTab);
  const setTrainTab = useTraining((s) => s.setTrainTab);

  return (
    <div className="mx-auto max-w-[1180px] px-7 pt-6 pb-[120px]">
      <Tabs
        value={trainTab}
        onChange={(v) => setTrainTab(v as TrainTab)}
        options={TABS}
        className="mb-[22px]"
      />

      <div className="grid grid-cols-[1fr_300px] items-start gap-6">
        <div className="flex min-w-0 flex-col gap-3.5">
          {trainTab === "styletts" ? (
            <StyleTtsForm />
          ) : (
            <SmallModelForm variant={trainTab} />
          )}
        </div>
        <div className="sticky top-0 flex flex-col gap-3.5">
          <QueueCard />
        </div>
      </div>
    </div>
  );
}
