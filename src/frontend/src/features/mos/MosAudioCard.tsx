import { backendResourceUrl } from "@/app/backend";
import { AudioScoreInput } from "@/features/audio/AudioScoreInput";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Card } from "@/shared/ui/Card";
import { cn } from "@/shared/ui/cn";
import type { MosAudio } from "./api";
import { audioSeed, pairScoreDraft } from "./logic";

export function MosAudioCard({
  label,
  audio,
  score,
  preferred,
  disabled,
  onScore,
  onPreferred,
}: {
  label: string;
  audio: MosAudio;
  score: string;
  preferred: boolean;
  disabled: boolean;
  onScore: (value: string) => void;
  onPreferred: () => void;
}) {
  const contentUrl = backendResourceUrl(`/audio-files/${encodeURIComponent(audio.id)}/content`);
  return (
    <Card className={cn("overflow-hidden transition-colors", preferred && "border-blue-500 bg-blue-50")}>
      <div className="flex items-start justify-between gap-3 border-b border-line px-4 py-3.5">
        <div className="min-w-0">
          <div className="text-[11px] font-bold uppercase tracking-wider text-blue-600">Sample {label}</div>
          <div className="truncate text-[14px] font-bold text-txt">{audio.name}</div>
          <div className="truncate text-[12px] text-txt-mute">{audio.speaker || "No speaker"}</div>
        </div>
        <AudioScoreInput
          value={score}
          disabled={disabled}
          label="MOS"
          onChange={onScore}
          onCancel={() => onScore(pairScoreDraft(audio.score))}
        />
      </div>
      <div className="px-4 py-4">
        <WaveformPlayer seed={audioSeed(audio.id)} duration={audio.duration} fileName={audio.name} src={contentUrl} />
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={onPreferred}
        className={cn(
          "flex h-11 w-full items-center justify-center border-0 text-[13px] font-bold transition-colors",
          preferred ? "bg-blue-500 text-white" : "bg-panel-2 text-txt-dim hover:bg-panel-3",
        )}
      >
        {preferred ? "Selected as better" : "Choose as better"}
      </button>
    </Card>
  );
}
