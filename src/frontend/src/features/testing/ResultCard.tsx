import { backendResourceUrl } from "@/app/backend";
import { fmtAgo } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Badge } from "@/shared/ui/Badge";
import { Card } from "@/shared/ui/Card";
import type { TestResult } from "./store";

export function ResultCard({ result }: { result: TestResult }) {
  const badge =
    result.state === "succeeded"
      ? { tone: "emerald" as const, label: "SYNTHESIZED" }
      : result.state === "failed"
        ? { tone: "red" as const, label: "FAILED" }
        : { tone: "blue" as const, label: "SYNTHESIZING…" };
  const src = result.audioFileId
    ? backendResourceUrl(`/audio-files/${encodeURIComponent(result.audioFileId)}/content`)
    : undefined;

  return (
    <Card className="min-w-0 px-[18px] py-4">
      <div className="mb-2.5 flex flex-wrap items-center gap-2">
        <Badge tone={badge.tone} shape="pill">
          {badge.label}
        </Badge>
        <span className="font-mono text-[11px] text-txt-mute">
          {result.steps} steps · scale {result.emb.toFixed(1)}
        </span>
        <div className="flex-1" />
        <span className="text-[11px] text-txt-mute">{fmtAgo(result.when)}</span>
      </div>
      <div className="mb-3.5 text-[13.5px] leading-relaxed text-txt">{result.text}</div>
      {result.state === "succeeded" && result.audioFileId ? (
        <WaveformPlayer seed={result.id.length} duration={result.duration ?? 0} fileName={result.name ?? "synthesis.wav"} src={src} />
      ) : result.state === "failed" ? (
        <div className="flex items-start gap-2 rounded-md bg-red-50 px-3 py-2.5 text-xs font-semibold text-red-700">
          <Icon name="alert" size={14} className="mt-px text-red-600" />
          <span className="break-words">{result.error ?? "Synthesis failed"}</span>
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs font-semibold text-txt-mute">
          <Icon name="loader" size={14} className="animate-spin text-blue-500" />
          Running on the runner…
        </div>
      )}
    </Card>
  );
}
