import { ArrowUpRight } from "lucide-react";

import type { Run } from "@/shared/types";
import { Button, Caption, Sheet, StatusBadge, cn } from "@/shared/ui";

import { ancestorChain, formatBytes, type PlacedCheckpoint } from "./logic";

interface CheckpointInspectorProps {
  checkpoint: PlacedCheckpoint;
  nodes: PlacedCheckpoint[];
  runNames: Map<string, string>;
  run: Run | undefined;
  onOpenRun: (runId: string, additive: boolean) => void;
  onSelect: (checkpointId: string) => void;
  onClose: () => void;
}

export function CheckpointInspector({
  checkpoint,
  nodes,
  runNames,
  run,
  onOpenRun,
  onSelect,
  onClose,
}: CheckpointInspectorProps) {
  const chain = ancestorChain(checkpoint.id, nodes);
  const children = nodes.filter((node) => node.ancestorId === checkpoint.id);

  return (
    <Sheet open onClose={onClose} title={`step ${checkpoint.step.toLocaleString()}`} width={400}>
      <div className="flex flex-col gap-4 p-3">
        <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
          <Caption>Run</Caption>
          <dd className="m-0 flex min-w-0 items-center gap-2">
            <span className="truncate text-fg">{runNames.get(checkpoint.runId) ?? checkpoint.runId}</span>
            {run === undefined ? null : <StatusBadge status={run.status} />}
          </dd>
          <Caption>Checkpoint</Caption>
          <dd className="m-0 truncate font-mono text-fg-secondary" title={checkpoint.name}>{checkpoint.name}</dd>
          <Caption>Created</Caption>
          <dd className="m-0 font-mono text-fg-secondary">{new Date(checkpoint.createdAt).toLocaleString()}</dd>
          <Caption>Size</Caption>
          <dd className="m-0 font-mono tabular-nums text-fg-secondary">{formatBytes(checkpoint.sizeBytes)}</dd>
          <Caption>Type</Caption>
          <dd className="m-0 font-mono text-fg-secondary">{checkpoint.type}</dd>
          <Caption>Asset id</Caption>
          <dd className="m-0 truncate font-mono text-fg-muted" title={checkpoint.id}>{checkpoint.id}</dd>
        </dl>

        {run === undefined ? null : (
          <Button variant="secondary" onClick={() => onOpenRun(checkpoint.runId, false)}>
            <ArrowUpRight size={13} />
            Show this run
          </Button>
        )}

        <section className="flex flex-col gap-1">
          <Caption>Resume chain</Caption>
          <ol className="m-0 flex list-none flex-col gap-0.5 p-0">
            {chain.map((node) => (
              <li key={node.id}>
                <ChainRow
                  node={node}
                  runName={runNames.get(node.runId) ?? node.runId}
                  current={node.id === checkpoint.id}
                  onSelect={onSelect}
                />
              </li>
            ))}
          </ol>
        </section>

        {children.length === 0 ? null : (
          <section className="flex flex-col gap-1">
            <Caption>Resumed by</Caption>
            <ul className="m-0 flex list-none flex-col gap-0.5 p-0">
              {children.map((node) => (
                <li key={node.id}>
                  <ChainRow
                    node={node}
                    runName={runNames.get(node.runId) ?? node.runId}
                    current={false}
                    onSelect={onSelect}
                  />
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </Sheet>
  );
}

interface ChainRowProps {
  node: PlacedCheckpoint;
  runName: string;
  current: boolean;
  onSelect: (checkpointId: string) => void;
}

function ChainRow({ node, runName, current, onSelect }: ChainRowProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(node.id)}
      className={cn(
        "flex w-full min-w-0 items-baseline gap-2 rounded-sm px-1.5 py-1 text-left text-xs hover:bg-hover",
        current ? "bg-selected text-fg" : "text-fg-secondary",
      )}
    >
      <span className="shrink-0 font-mono tabular-nums">{node.step.toLocaleString()}</span>
      <span className="truncate">{runName}</span>
    </button>
  );
}
