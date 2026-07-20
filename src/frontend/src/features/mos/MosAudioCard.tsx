import { backendResourceUrl } from "@/app/backend";
import { AudioScoreInput } from "@/features/audio/AudioScoreInput";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Card } from "@/shared/ui/Card";
import type { MosAudio } from "./api";
import { audioSeed, pairScoreDraft } from "./logic";

export function MosAudioCard({
  label,
  audio,
  score,
  disabled,
  onScore,
  onChoose,
}: {
  label: string;
  audio: MosAudio;
  score: string;
  disabled: boolean;
  onScore: (value: string) => void;
  onChoose: () => void;
}) {
  const contentUrl = backendResourceUrl(`/audio-files/${encodeURIComponent(audio.id)}/content`);
  return (
    <Card className="overflow-hidden">
      <div className="flex items-start justify-between gap-3 border-b border-line px-4 py-3.5">
        <div className="min-w-0">
          <div className="text-[11px] font-bold uppercase tracking-wider text-blue-600">Sample {label}</div>
          <div className="truncate text-[14px] font-bold text-txt">{audio.name}</div>
          <div className="truncate text-[12px] text-txt-mute">{audio.annotations.speaker_id || "No speaker"}</div>
        </div>
        <AudioScoreInput
          value={score}
          disabled={disabled}
          label="MOS"
          onChange={onScore}
          onCancel={() => onScore(pairScoreDraft(audio.annotations.score))}
        />
      </div>
      <div className="px-4 py-4">
        <WaveformPlayer seed={audioSeed(audio.id)} duration={audio.duration} fileName={audio.name} src={contentUrl} />
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={onChoose}
        className="flex h-11 w-full items-center justify-center border-0 bg-blue-500 text-[13px] font-bold text-white transition-colors hover:bg-blue-600 disabled:bg-panel-2 disabled:text-txt-mute"
      >
        Choose {label} as better and save
      </button>
    </Card>
  );
}
