import { Icon } from "@/shared/icons";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { IconButton } from "@/shared/ui/IconButton";
import { type Artifact, artifactContentUrl } from "./api";
import { useArtifactActions, useArtifactsQuery } from "./query";

export function ArtifactsScreen() {
  const artifacts = useArtifactsQuery();
  const actions = useArtifactActions();
  const rows = artifacts.data ?? [];

  return (
    <div className="mx-auto max-w-[1180px] px-7 pb-16 pt-5">
      <div className="mb-4 flex items-center gap-2 text-xs font-semibold text-txt-dim">
        <Icon name="sparkles" size={14} strokeWidth={2} className="text-txt-mute" />
        <span>Generated plots and figures produced by workflow runs</span>
        {rows.length ? <span className="font-mono text-[10px] text-txt-mute">{rows.length}</span> : null}
      </div>

      {artifacts.isLoading ? (
        <Card className="p-6 text-sm text-txt-mute">Loading artifacts...</Card>
      ) : artifacts.isError ? (
        <Card>
          <EmptyState icon="alert" title="Couldn't reach the backend" description="The artifacts service didn't respond." />
        </Card>
      ) : rows.length ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(340px,1fr))] gap-4">
          {rows.map((artifact) => (
            <ArtifactCard key={artifact.id} artifact={artifact} onDelete={() => actions.remove(artifact.id)} />
          ))}
        </div>
      ) : (
        <Card>
          <EmptyState
            icon="sparkles"
            title="No artifacts yet"
            description="Run a workflow with a plotting node (e.g. Embed Voices PCA) to produce figures here."
          />
        </Card>
      )}
    </div>
  );
}

function ArtifactCard({ artifact, onDelete }: { artifact: Artifact; onDelete: () => void }) {
  const url = artifactContentUrl(artifact.id);
  const voices = artifact.metadata["voices"];
  const pointCount = artifact.metadata["point_count"];
  const contentType = String(artifact.metadata["content_type"] ?? "");
  const isHtml = contentType.startsWith("text/html");

  return (
    <Card className="flex flex-col overflow-hidden">
      {isHtml ? (
        <div className="relative h-[300px] w-full bg-panel-2">
          <iframe src={url} title={artifact.name} className="h-full w-full border-0" sandbox="allow-scripts" />
          <a href={url} target="_blank" rel="noreferrer" className="absolute right-2 top-2 rounded-md bg-black/60 px-2 py-1 text-[11px] font-semibold text-white hover:bg-black/80">
            Open ↗
          </a>
        </div>
      ) : (
        <a href={url} target="_blank" rel="noreferrer" className="block bg-panel-2" title="Open full size">
          <img src={url} alt={artifact.name} className="h-[240px] w-full object-contain" />
        </a>
      )}
      <div className="flex items-start gap-2 p-3">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-bold text-txt" title={artifact.name}>{artifact.name}</div>
          <div className="mt-0.5 font-mono text-[11px] text-txt-mute">
            {Array.isArray(voices) ? `${voices.length} voices` : "figure"}
            {typeof pointCount === "number" ? ` · ${pointCount} points` : ""}
            {` · ${formatBytes(artifact.size)}`}
          </div>
        </div>
        <IconButton icon="trash" danger title="Delete artifact" onClick={onDelete} />
      </div>
    </Card>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
