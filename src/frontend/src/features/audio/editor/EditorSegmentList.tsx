import { VirtualTable } from "@/shared/data/VirtualTable";
import { Button } from "@/shared/ui/Button";
import { SearchInput } from "@/shared/ui/SearchInput";
import { SegmentRow } from "../SegmentRow";
import { useEditor } from "../editorStore";

export function EditorSegmentList() {
  const { segs, segChecked, segQuery, setQuery, setChecked, deleteChecked, addSeg } = useEditor();
  const query = segQuery.trim().toLowerCase();
  const visible = query ? segs.filter((segment) => segment.text.toLowerCase().includes(query) || segment.phon.toLowerCase().includes(query)) : segs;
  const allVisibleChecked = visible.length > 0 && visible.every((segment) => segChecked.includes(segment.id));
  return (
    <div className="flex flex-col rounded-[10px] border border-line bg-panel p-4">
      <div className="mb-3 flex items-center gap-3">
        <label className="flex items-center gap-2" title={allVisibleChecked ? "Clear selection" : "Select all shown"}>
          <input
            type="checkbox"
            checked={allVisibleChecked}
            disabled={!visible.length}
            onChange={() => setChecked(allVisibleChecked ? [] : visible.map((segment) => segment.id))}
            className="h-3.5 w-3.5 cursor-pointer accent-blue-500 disabled:opacity-40"
          />
          <span className="text-[15px] font-bold">Segments <span className="font-semibold text-txt-mute">{query ? `(${visible.length} of ${segs.length})` : `(${segs.length})`}</span></span>
        </label>
        <SearchInput value={segQuery} onChange={setQuery} placeholder="Search transcripts / phonemes…" />
        <div className="flex-1" />
        {segChecked.length ? <Button variant="secondary" icon="trash" onClick={deleteChecked}>Delete {segChecked.length} selected</Button> : null}
        <Button variant="ghost" icon="plus" onClick={addSeg}>Add segment</Button>
      </div>
      {visible.length ? (
        <VirtualTable
          count={visible.length}
          estimateRowHeight={72}
          pageScroll
          renderRow={(index) => <SegmentRow seg={visible[index]!} index={segs.indexOf(visible[index]!)} isLast={segs.indexOf(visible[index]!) === segs.length - 1} />}
        />
      ) : (
        <div className="p-10 text-center text-[13px] text-txt-mute">{query ? `No segments match "${segQuery}".` : "No segments — add one to begin transcribing."}</div>
      )}
    </div>
  );
}
