import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { showToast } from "@/shared/feedback/Toast";
import { Icon } from "@/shared/icons";
import { useDeleteVoicesMutation } from "./query";
import { useVoiceFilters } from "./store";

export function VoiceSelectionBar({ total }: { total: number }) {
  const { query, selection, selectAllMatching, selectAllFiltered, clearSelection } = useVoiceFilters();
  const deleteVoices = useDeleteVoicesMutation();
  const ids = Object.keys(selection);
  const count = selectAllMatching ? total : ids.length;
  const label = `${count.toLocaleString()} voice${count === 1 ? "" : "s"}`;

  const deleteSelected = () =>
    askConfirm({
      title: "Delete voices?",
      desc: `Permanently delete ${label}. Segments keep their text but lose these voice labels. This cannot be undone.`,
      danger: true,
      label: "Delete voices",
      onConfirm: () => {
        const request = selectAllMatching
          ? { mode: "filter" as const, query }
          : { mode: "ids" as const, ids };
        deleteVoices.mutate(request, {
          onSuccess: () => {
            clearSelection();
            showToast(`Deleted ${label}`, undefined, "error");
          },
        });
      },
    });

  return (
    <div className="relative mb-3 flex items-center gap-2.5 rounded-[9px] bg-gray-900 py-2.5 pl-4 pr-3">
      <span className="text-[13px] font-bold text-white">{count.toLocaleString()} selected</span>
      {!selectAllMatching ? (
        <button onClick={selectAllFiltered} className="text-xs font-semibold text-blue-300">
          · Select all {total.toLocaleString()} matching filter
        </button>
      ) : null}
      <div className="flex-1" />
      <button
        disabled={deleteVoices.isPending || count === 0}
        onClick={deleteSelected}
        className="flex h-[34px] items-center gap-1.5 rounded-md bg-red-500 px-3 text-[12.5px] font-semibold text-white hover:bg-red-600 disabled:cursor-default disabled:opacity-60"
      >
        <Icon name="trash" size={15} strokeWidth={2.2} />
        Delete
      </button>
      <button
        onClick={clearSelection}
        title="Clear selection"
        className="flex h-[34px] w-[34px] items-center justify-center rounded-md bg-white/10 text-white hover:bg-white/15"
      >
        <Icon name="x" size={16} strokeWidth={2.4} />
      </button>
    </div>
  );
}
