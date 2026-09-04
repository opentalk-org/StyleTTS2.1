import { Images, Link2, Link2Off } from "lucide-react";
import { useMemo, useState } from "react";

import { useViewerStore } from "@/features/viewer/store";
import type { Artifact, PanelColumns, Run } from "@/shared/types";
import { Card, cn, Collapsible, EmptyState, IconButton, SegmentedControl, Toolbar } from "@/shared/ui";

import { COLUMN_CLASSES } from "./ChartSection";
import { sectionize } from "./logic";
import { ArtifactValue, ImageLightbox, kindIcon, StepControl } from "./MediaCard";

interface MediaPanelProps {
  runs: Run[];
  runColors: Record<string, string>;
  artifacts: Artifact[];
}

export function MediaPanel({ runs, runColors, artifacts }: MediaPanelProps) {
  const allSteps = useMemo(() => [...new Set(artifacts.map((artifact) => artifact.step))].sort((a, b) => a - b), [artifacts]);
  const [globalIndex, setGlobalIndex] = useState<number | null>(null);
  const columns = useViewerStore((state) => state.mediaColumns);
  const setColumns = useViewerStore((state) => state.setMediaColumns);
  const [collapsed, setCollapsed] = useState<string[]>([]);
  const byName = useMemo(() => {
    const groups = new Map<string, Artifact[]>();
    for (const artifact of artifacts) groups.set(artifact.name, [...(groups.get(artifact.name) ?? []), artifact]);
    return [...groups.entries()].map(([name, items]) => ({ name, items }));
  }, [artifacts]);
  const sections = useMemo(() => sectionize(byName, []), [byName]);
  const globalStep = allSteps.length === 0 ? null : allSteps[Math.min(globalIndex ?? allSteps.length - 1, allSteps.length - 1)];

  if (runs.length === 0) {
    return <EmptyState icon={<Images />} title="Select runs to see media" description="Images, audio, text and plot artifacts logged by the selected runs appear here." />;
  }
  if (artifacts.length === 0) {
    return <EmptyState icon={<Images />} title="No media logged" description="The selected runs have not logged any artifacts." />;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Toolbar
        start={
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <span className="shrink-0 text-xs text-fg-muted">Step for every group</span>
            <StepControl
              label="Every media group"
              steps={allSteps}
              index={Math.min(globalIndex ?? allSteps.length - 1, allSteps.length - 1)}
              onIndex={setGlobalIndex}
              className="max-w-md flex-1"
            />
          </div>
        }
        end={
          <SegmentedControl
            label="Media columns"
            value={columns}
            onValue={setColumns}
            options={[
              { value: "1" as const, label: "1" },
              { value: "2" as const, label: "2" },
              { value: "3" as const, label: "3" },
              { value: "auto" as const, label: "Auto" },
            ]}
          />
        }
      />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="flex flex-col gap-4 p-4">
          {sections.map((section) => {
            const open = !collapsed.includes(section.name);
            return (
              <Collapsible
                key={section.name}
                open={open}
                onToggle={() =>
                  setCollapsed((current) =>
                    current.includes(section.name) ? current.filter((item) => item !== section.name) : [...current, section.name],
                  )
                }
                title={section.name}
                count={section.items.length}
              >
                <div className="flex flex-col gap-3">
                  {section.items.map((group) => (
                    <ArtifactSeries key={group.name} name={group.name} runs={runs} runColors={runColors} artifacts={group.items} globalStep={globalStep} columns={columns} />
                  ))}
                </div>
              </Collapsible>
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface ArtifactSeriesProps {
  name: string;
  runs: Run[];
  runColors: Record<string, string>;
  artifacts: Artifact[];
  globalStep: number | null;
  columns: PanelColumns;
}

function ArtifactSeries({ name, runs, runColors, artifacts, globalStep, columns }: ArtifactSeriesProps) {
  const steps = useMemo(() => [...new Set(artifacts.map((artifact) => artifact.step))].sort((a, b) => a - b), [artifacts]);
  const [linked, setLinked] = useState(true);
  const [ownIndex, setOwnIndex] = useState(steps.length - 1);
  const [zoomedRun, setZoomedRun] = useState<Run | null>(null);
  const linkedIndex = globalStep === null ? steps.length - 1 : nearestIndex(steps, globalStep);
  const index = linked ? linkedIndex : Math.min(ownIndex, steps.length - 1);
  const step = steps[index];
  const kind = artifacts[0].kind;

  function artifactFor(run: Run, atStep: number): Artifact | undefined {
    return artifacts.find((artifact) => artifact.runId === run.id && artifact.step === atStep);
  }

  return (
    <Card>
      <div className="flex min-h-card-head flex-wrap items-center gap-x-3 gap-y-2 border-b border-line py-1 pr-1 pl-3">
        <span className="flex min-w-0 items-center gap-2 text-fg-secondary">
          {kindIcon(kind)}
          <span className="truncate text-[13px] font-medium text-fg" title={name}>{name.split("/").at(-1)}</span>
        </span>
        <span className={cn("flex min-w-0 flex-1 items-center gap-1", linked ? "opacity-70" : "")}>
          <StepControl label={name} steps={steps} index={index} onIndex={(next) => { setLinked(false); setOwnIndex(next); }} className="max-w-sm flex-1" />
          <IconButton label={linked ? "Following the step above. Click to unlink" : "Own step. Click to follow the step above"} size="sm" active={linked} onClick={() => setLinked(!linked)}>
            {linked ? <Link2 size={12} /> : <Link2Off size={12} />}
          </IconButton>
        </span>
      </div>
      <div className={cn("grid gap-3 p-2", COLUMN_CLASSES[columns])}>
        {runs.map((run) => (
          <ArtifactValue key={run.id} run={run} color={runColors[run.id]} kind={kind} artifact={artifactFor(run, step)} onZoom={() => setZoomedRun(run)} />
        ))}
      </div>
      {zoomedRun === null ? null : (
        <ImageLightbox
          run={zoomedRun}
          name={name}
          artifact={artifactFor(zoomedRun, step)}
          steps={steps}
          index={index}
          onIndex={(next) => { setLinked(false); setOwnIndex(next); }}
          onClose={() => setZoomedRun(null)}
        />
      )}
    </Card>
  );
}

function nearestIndex(steps: number[], target: number): number {
  let best = 0;
  steps.forEach((step, index) => {
    if (step <= target) best = index;
  });
  return best;
}
