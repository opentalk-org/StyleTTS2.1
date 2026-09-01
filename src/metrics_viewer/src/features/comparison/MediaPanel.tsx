import {
  ChevronLeft,
  ChevronRight,
  FileAudio,
  FileChartColumn,
  FileImage,
  FileText,
  Maximize2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import type { Artifact, ArtifactKind, Run } from "@/shared/types";
import { AudioPlayer, Card, CardHeader, GroupLabel, IconButton, Modal, Range } from "@/shared/ui";

export function MediaPanel({ runs, artifacts }: { runs: Run[]; artifacts: Artifact[] }) {
  const names = useMemo(
    () => [...new Set(artifacts.map((artifact) => artifact.name))].sort(),
    [artifacts],
  );
  if (names.length === 0) return null;
  return (
    <div className="flex flex-col gap-3">
      {names.map((name) => (
        <ArtifactSeries
          key={name}
          name={name}
          runs={runs}
          artifacts={artifacts.filter((artifact) => artifact.name === name)}
        />
      ))}
    </div>
  );
}

function ArtifactSeries({ name, runs, artifacts }: { name: string; runs: Run[]; artifacts: Artifact[] }) {
  const steps = useMemo(
    () => [...new Set(artifacts.map((artifact) => artifact.step))].sort((a, b) => a - b),
    [artifacts],
  );
  const [stepIndex, setStepIndex] = useState(steps.length - 1);
  const [zoomedRun, setZoomedRun] = useState<Run | null>(null);
  const safeIndex = Math.min(stepIndex, steps.length - 1);
  const step = steps[safeIndex];
  const kind = artifacts[0].kind;

  function artifactFor(run: Run, atStep: number): Artifact | undefined {
    return artifacts.find((artifact) => artifact.runId === run.id && artifact.step === atStep);
  }

  return (
    <Card>
      <CardHeader className="h-auto min-h-14 flex-wrap gap-y-2 py-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="grid size-7 shrink-0 place-items-center rounded-lg border border-line bg-inset text-fg-secondary">
            {kindIcon(kind)}
          </span>
          <span className="flex min-w-0 flex-col">
            <strong className="truncate text-sm font-medium text-fg">{entryName(name)}</strong>
            <span className="truncate font-mono text-[10px] text-fg-muted">{name}</span>
          </span>
        </div>
        <StepControl
          label={name}
          steps={steps}
          index={safeIndex}
          onIndex={setStepIndex}
          className="w-full max-w-[420px] sm:w-auto sm:flex-1"
        />
      </CardHeader>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(250px,1fr))] gap-2 p-2">
        {runs.map((run) => (
          <ArtifactValue
            key={run.id}
            run={run}
            kind={kind}
            artifact={artifactFor(run, step)}
            onZoom={() => setZoomedRun(run)}
          />
        ))}
      </div>

      {zoomedRun === null ? null : (
        <ImageLightbox
          run={zoomedRun}
          name={name}
          artifact={artifactFor(zoomedRun, step)}
          steps={steps}
          index={safeIndex}
          onIndex={setStepIndex}
          onClose={() => setZoomedRun(null)}
        />
      )}
    </Card>
  );
}

interface StepControlProps {
  label: string;
  steps: number[];
  index: number;
  onIndex: (index: number) => void;
  className?: string;
}

/** Prev / slider / next over the logged steps, shared by the card and the lightbox. */
function StepControl({ label, steps, index, onIndex, className }: StepControlProps) {
  return (
    <div className={`flex min-w-0 items-center gap-2 ${className ?? ""}`}>
      <IconButton
        label="Previous step"
        size="sm"
        variant="secondary"
        disabled={index === 0}
        onClick={() => onIndex(Math.max(0, index - 1))}
      >
        <ChevronLeft size={13} />
      </IconButton>
      <Range
        aria-label={`${label} step`}
        min={0}
        max={steps.length - 1}
        step={1}
        value={index}
        onValue={onIndex}
        className="min-w-0 flex-1"
      />
      <IconButton
        label="Next step"
        size="sm"
        variant="secondary"
        disabled={index === steps.length - 1}
        onClick={() => onIndex(Math.min(steps.length - 1, index + 1))}
      >
        <ChevronRight size={13} />
      </IconButton>
      <span className="shrink-0 font-mono text-xs tabular-nums whitespace-nowrap text-fg-muted">
        step <span className="text-fg">{steps[index].toLocaleString()}</span>
      </span>
    </div>
  );
}

function ArtifactValue({
  run,
  kind,
  artifact,
  onZoom,
}: {
  run: Run;
  kind: ArtifactKind;
  artifact: Artifact | undefined;
  onZoom: () => void;
}) {
  return (
    <article className="group/artifact min-w-0 overflow-hidden rounded-lg border border-line bg-inset">
      <header className="border-b border-line px-2.5 py-1.5">
        <GroupLabel className="truncate">{run.name}</GroupLabel>
      </header>
      {artifact === undefined ? (
        <p className="m-0 px-3 py-6 text-center text-xs text-fg-muted">Not logged at this step</p>
      ) : kind === "audio" ? (
        <AudioPlayer src={artifact.source} label={`${run.name} ${entryName(artifact.name)}`} />
      ) : kind === "image" ? (
        <button
          type="button"
          onClick={onZoom}
          title="Open full size"
          className="relative block w-full cursor-zoom-in"
        >
          <img
            loading="lazy"
            src={artifact.source}
            alt={`${run.name} ${artifact.name}`}
            className="block h-44 w-full bg-deep object-contain"
          />
          <span className="absolute top-2 right-2 grid size-6 place-items-center rounded-md border border-line-hover bg-deep/80 text-fg-secondary opacity-0 transition-opacity duration-150 group-hover/artifact:opacity-100">
            <Maximize2 size={12} />
          </span>
        </button>
      ) : kind === "plot" ? (
        <MiniPlot values={JSON.parse(artifact.source) as number[]} />
      ) : (
        <pre className="m-0 p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-fg-secondary">
          {artifact.source}
        </pre>
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

/** Full-viewport view of one run's image, with the step control kept in reach. */
function ImageLightbox({ run, name, artifact, steps, index, onIndex, onClose }: ImageLightboxProps) {
  const frameRef = useRef<HTMLDivElement>(null);

  // autoFocus applies to form controls only, so the key handler is focused here.
  useEffect(() => {
    frameRef.current?.focus();
  }, []);

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      onIndex(Math.max(0, index - 1));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      onIndex(Math.min(steps.length - 1, index + 1));
    }
  }

  return (
    <Modal open onClose={onClose} label={`${run.name} ${entryName(name)}`}>
      {/* Arrow keys step through without having to aim at the control. */}
      <div
        ref={frameRef}
        className="flex min-h-0 flex-1 flex-col focus:outline-none"
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <header className="flex h-14 flex-none items-center justify-between gap-3 border-b border-line px-4">
          <div className="flex min-w-0 flex-col gap-1">
            <GroupLabel>{run.name}</GroupLabel>
            <h2 className="m-0 truncate text-base leading-tight font-semibold tracking-tight text-fg">
              {entryName(name)}
            </h2>
          </div>
          <IconButton label="Close image" onClick={onClose}>
            <X size={15} />
          </IconButton>
        </header>

        <div className="grid min-h-0 flex-1 place-items-center overflow-auto bg-deep p-4">
          {artifact === undefined ? (
            <p className="m-0 text-xs text-fg-muted">Not logged at this step</p>
          ) : (
            <img
              src={artifact.source}
              alt={`${run.name} ${name}`}
              className="max-h-full max-w-full object-contain"
            />
          )}
        </div>

        <footer className="flex flex-none items-center justify-between gap-3 border-t border-line p-3">
          <span className="font-mono text-[11px] text-fg-muted">{name}</span>
          <StepControl
            label={name}
            steps={steps}
            index={index}
            onIndex={onIndex}
            className="w-full max-w-md"
          />
        </footer>
      </div>
    </Modal>
  );
}

function kindIcon(kind: ArtifactKind): ReactNode {
  if (kind === "audio") return <FileAudio size={14} />;
  if (kind === "image") return <FileImage size={14} />;
  if (kind === "plot") return <FileChartColumn size={14} />;
  return <FileText size={14} />;
}

function entryName(name: string): string {
  return name.split("/").at(-1) ?? name;
}

function MiniPlot({ values }: { values: number[] }) {
  const max = Math.max(...values);
  return (
    <div className="flex h-44 items-end gap-[3px] px-3.5 pt-5 pb-3" aria-label="Plot artifact preview">
      {values.map((value, index) => (
        <i
          key={index}
          style={{ height: `${(value / max) * 100}%` }}
          title={`bin ${index}: ${value.toFixed(2)}`}
          className="flex-1 rounded-t-[2px] bg-accent/70 transition-colors duration-150 hover:bg-accent-bright"
        />
      ))}
    </div>
  );
}
