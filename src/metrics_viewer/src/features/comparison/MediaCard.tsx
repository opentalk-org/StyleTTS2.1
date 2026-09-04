import { ChevronLeft, ChevronRight, FileAudio, FileChartColumn, FileImage, FileText, Maximize2 } from "lucide-react";
import { useEffect, type ReactNode } from "react";

import type { Artifact, ArtifactKind, Run } from "@/shared/types";
import { AudioPlayer, cn, Dialog, IconButton, Range } from "@/shared/ui";

const MEDIA_CONTENT_HEIGHT = 200;

export interface StepControlProps {
  label: string;
  steps: number[];
  index: number;
  onIndex: (index: number) => void;
  className?: string;
}

export function StepControl({ label, steps, index, onIndex, className }: StepControlProps) {
  return (
    <div className={cn("flex min-w-0 items-center gap-1", className)}>
      <IconButton label="Previous step" size="sm" disabled={index === 0} onClick={() => onIndex(Math.max(0, index - 1))}>
        <ChevronLeft size={13} />
      </IconButton>
      <Range aria-label={`${label} step`} min={0} max={steps.length - 1} step={1} value={index} onValue={onIndex} className="min-w-0 flex-1" />
      <IconButton label="Next step" size="sm" disabled={index === steps.length - 1} onClick={() => onIndex(Math.min(steps.length - 1, index + 1))}>
        <ChevronRight size={13} />
      </IconButton>
      <span className="w-20 shrink-0 text-right font-mono text-xs tabular-nums text-fg-muted">
        step <span className="text-fg">{steps[index].toLocaleString()}</span>
      </span>
    </div>
  );
}

interface ArtifactValueProps {
  run: Run;
  color: string;
  kind: ArtifactKind;
  artifact: Artifact | undefined;
  onZoom: () => void;
}

export function ArtifactValue({ run, color, kind, artifact, onZoom }: ArtifactValueProps) {
  return (
    <article className="group/artifact min-w-0 overflow-hidden rounded-md border border-line bg-inset">
      <header className="flex items-center gap-1.5 border-b border-line px-2 py-1">
        <span className="size-2 shrink-0 rounded-full" style={{ background: color }} />
        <span className="truncate text-xs font-medium text-fg" title={run.name}>{run.name}</span>
      </header>
      {artifact === undefined ? (
        <p className="grid place-items-center px-3 text-center text-xs text-fg-muted" style={{ height: MEDIA_CONTENT_HEIGHT }}>
          Not logged at this step
        </p>
      ) : kind === "audio" ? (
        <div className="flex items-center px-3" style={{ height: MEDIA_CONTENT_HEIGHT }}>
          <AudioPlayer src={artifact.source} label={`${run.name} ${artifact.name}`} />
        </div>
      ) : kind === "image" ? (
        <button type="button" onClick={onZoom} title="Open full size" className="relative block w-full cursor-zoom-in">
          <img loading="lazy" src={artifact.source} alt={`${run.name} ${artifact.name}`} className="block w-full bg-canvas object-contain" style={{ height: MEDIA_CONTENT_HEIGHT }} />
          <span className="absolute top-1.5 right-1.5 grid size-6 place-items-center rounded-sm border border-line bg-raised text-fg-secondary opacity-0 group-hover/artifact:opacity-100">
            <Maximize2 size={12} />
          </span>
        </button>
      ) : kind === "plot" ? (
        <MiniPlot values={JSON.parse(artifact.source) as number[]} color={color} />
      ) : (
        <pre className="m-0 overflow-auto p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-fg-secondary" style={{ height: MEDIA_CONTENT_HEIGHT }}>{artifact.source}</pre>
      )}
    </article>
  );
}

interface ImageLightboxProps {
  run: Run;
  name: string;
  artifact: Artifact | undefined;
  steps: number[];
  index: number;
  onIndex: (index: number) => void;
  onClose: () => void;
}

export function ImageLightbox({ run, name, artifact, steps, index, onIndex, onClose }: ImageLightboxProps) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "ArrowLeft") onIndex(Math.max(0, index - 1));
      if (event.key === "ArrowRight") onIndex(Math.min(steps.length - 1, index + 1));
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [index, steps.length, onIndex]);

  return (
    <Dialog
      open
      onClose={onClose}
      eyebrow={run.name}
      title={name}
      size="viewport"
      footer={<StepControl label={name} steps={steps} index={index} onIndex={onIndex} className="w-full max-w-md" />}
    >
      <div className="grid min-h-0 flex-1 place-items-center overflow-auto bg-canvas p-4">
        {artifact === undefined ? (
          <p className="text-xs text-fg-muted">Not logged at this step</p>
        ) : (
          <img src={artifact.source} alt={`${run.name} ${name}`} className="h-full w-full object-contain" />
        )}
      </div>
    </Dialog>
  );
}

export function kindIcon(kind: ArtifactKind): ReactNode {
  if (kind === "audio") return <FileAudio size={14} />;
  if (kind === "image") return <FileImage size={14} />;
  if (kind === "plot") return <FileChartColumn size={14} />;
  return <FileText size={14} />;
}

function MiniPlot({ values, color }: { values: number[]; color: string }) {
  const max = Math.max(...values);
  return (
    <div className="flex items-end gap-[2px] px-3 pt-4 pb-3" style={{ height: MEDIA_CONTENT_HEIGHT }} aria-label="Plot artifact preview">
      {values.map((value, index) => (
        <i
          key={index}
          style={{ height: `${(value / max) * 100}%`, background: color }}
          title={`bin ${index}: ${value.toFixed(2)}`}
          className="flex-1 rounded-t-[2px] opacity-70 hover:opacity-100"
        />
      ))}
    </div>
  );
}
