import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { IconButton } from "@/shared/ui/IconButton";
import type { Dataset } from "@/mock/types";

export function DatasetCard({
  dataset,
  onOpen,
  onDelete,
}: {
  dataset: Dataset;
  onOpen: () => void;
  onDelete: (id: string) => void;
}) {
  const del = () =>
    askConfirm({
      title: "Delete dataset?",
      desc: `Delete "${dataset.name}". Files stay in the library but are unassigned from this dataset.`,
      danger: true,
      label: "Delete dataset",
      onConfirm: () => {
        onDelete(dataset.id);
        showToast("Dataset deleted", undefined, "error");
      },
    });

  return (
    <Card className="flex flex-col gap-3.5 p-[18px]">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 flex-none items-center justify-center rounded-[9px] bg-blue-50 text-blue-600">
          <Icon name="database" size={20} strokeWidth={2.2} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[15px] font-bold text-txt">{dataset.name}</div>
          <div className="mt-0.5 text-xs text-txt-mute">
            {dataset.id} · {dataset.files} files
          </div>
        </div>
      </div>
      <div className="flex gap-2">
        <Button variant="primary" icon="folder-open" className="flex-1" onClick={onOpen}>
          Open
        </Button>
        <IconButton
          icon="download"
          size={34}
          className="bg-panel-2"
          title="Export ZIP"
          onClick={() => showToast(`Exporting ${dataset.name}.zip…`, dataset.id)}
        />
        <IconButton icon="trash" size={34} danger className="bg-panel-2" title="Delete" onClick={del} />
      </div>
    </Card>
  );
}
