import { fmtAgo } from "@/shared/format";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Badge } from "@/shared/ui/Badge";
import { Card } from "@/shared/ui/Card";
import type { TestResult } from "./store";

/** One synthesized single-mode result: metadata header, text, phonemes, player. */
export function ResultCard({ result }: { result: TestResult }) {
  return (
    <Card className="min-w-0 px-[18px] py-4">
      <div className="mb-2.5 flex flex-wrap items-center gap-2">
        <Badge tone="emerald" shape="pill">
          SYNTHESIZED
        </Badge>
        <span className="font-mono text-[11px] text-txt-mute">
          {result.id} · {result.steps} steps · scale {result.emb.toFixed(1)}
        </span>
        <div className="flex-1" />
        <span className="text-[11px] text-txt-mute">{fmtAgo(result.when)}</span>
      </div>
      <div className="mb-1.5 text-[13.5px] leading-relaxed text-txt">{result.text}</div>
      <div className="mb-3.5 break-words font-mono text-xs leading-relaxed text-blue-600">
        {result.phon}
      </div>
      <WaveformPlayer seed={result.id.length} duration={result.dur} fileName={result.file} />
    </Card>
  );
}
