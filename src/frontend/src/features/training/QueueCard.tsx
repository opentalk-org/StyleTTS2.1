import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Card } from "@/shared/ui/Card";

import { useTraining } from "./store";

const BASE_SYMBOLS = 178;

type ValidationRow = { ok: boolean; label: string };

/** Sticky right rail: draft state, validation checklist, and the queue action. */
export function QueueCard() {
  const alphabet = useTraining((s) => s.alphabet);
  const oodSets = useTraining((s) => s.oodSets);

  const alphabetValid =
    alphabet.trim().split(/\s+/).filter(Boolean).length === BASE_SYMBOLS;

  const rows: ValidationRow[] = [
    { ok: true, label: "Display name set" },
    { ok: true, label: "Training dataset selected" },
    { ok: false, label: "Base checkpoint required" },
    { ok: alphabetValid, label: "Phoneme alphabet valid" },
    { ok: oodSets.length > 0, label: "OOD reference set added" },
  ];

  const queue = () => {
    const id = `job_${Math.random().toString(16).slice(2, 6)}`;
    showToast("Training job queued", id);
  };

  return (
    <Card className="p-[18px]">
      <div className="mb-3.5 flex items-center justify-between">
        <div className="text-sm font-bold text-txt">Queue training</div>
        <span className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-700">
          <Icon name="check" size={13} strokeWidth={2.5} className="text-emerald-600" />
          Draft saved
        </span>
      </div>

      <div className="mb-4 flex flex-col gap-2">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-2 text-[12.5px]">
            <Icon
              name={r.ok ? "check-circle" : "alert"}
              size={16}
              strokeWidth={2.2}
              className={r.ok ? "text-emerald-600" : "text-amber-600"}
            />
            <span
              className={
                r.ok ? "font-medium text-txt-dim" : "font-semibold text-amber-700"
              }
            >
              {r.label}
            </span>
          </div>
        ))}
      </div>

      <button
        onClick={queue}
        className="flex h-11 w-full items-center justify-center gap-2 rounded-lg border-0 bg-blue-500 text-sm font-semibold text-white cursor-pointer transition-colors hover:bg-blue-600"
      >
        <Icon name="bolt" size={17} strokeWidth={2.2} />
        Queue training job
      </button>

      <p className="mt-2.5 text-[11px] leading-relaxed text-txt-mute">
        Drafts autosave to this browser. Queuing returns a job id.
      </p>
    </Card>
  );
}
