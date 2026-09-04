import { useEffect, useRef, useState } from "react";

import { showToast } from "@/shared/feedback/Toast";
import { fmtAgo, fmtDur } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { WaveformBars } from "@/shared/media/WaveformBars";
import { cn } from "@/shared/ui/cn";
import { IconButton } from "@/shared/ui/IconButton";
import { InlineSegments } from "./InlineSegments";
import type { AudioFile } from "./api";
import { useAudioSegmentPreviewQuery, useRenameAudioFileMutation } from "./query";
import { useAudio } from "./store";
import { useNavigate } from "react-router-dom";

export const AUDIO_COLS = "30px 22px 66px minmax(180px,1.2fr) 54px 46px 74px 30px";

function Checkbox({ on }: { on: boolean }) {
  return (
    <span
      className={cn(
        "flex h-[18px] w-[18px] flex-none items-center justify-center rounded",
        on ? "bg-blue-500" : "border-2 border-line-2 bg-panel",
      )}
    >
      {on ? <Icon name="check" size={12} strokeWidth={3} className="text-white" /> : null}
    </span>
  );
}

export function AudioRow({ file, index }: { file: AudioFile; index: number }) {
  const { selection, selectAllMatching, expanded, toggleSelect, toggleExpanded } = useAudio();
  const on = selectAllMatching || !!selection[file.id];
  const ex = !!expanded[file.id];
  const noSeg = file.segments === 0;
  const updatedAt = Date.parse(file.updated_at);
  const navigate = useNavigate();
  const renameAudio = useRenameAudioFileMutation();
  const preview = useAudioSegmentPreviewQuery(file.id, ex);
  const skipNameBlur = useRef(false);
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState(file.name);

  useEffect(() => {
    if (!renaming) setNameDraft(file.name);
  }, [file.name, renaming]);

  const cancelRename = () => {
    skipNameBlur.current = true;
    setNameDraft(file.name);
    setRenaming(false);
  };

  const commitRename = async () => {
    const name = nameDraft.trim();
    if (!name) {
      showToast("Audio name is required", undefined, "error");
      return;
    }
    if (name === file.name) {
      setRenaming(false);
      return;
    }
    try {
      await renameAudio.mutateAsync({ id: file.id, name });
      setRenaming(false);
      showToast("Audio renamed");
    } catch {
      showToast("Could not rename audio", undefined, "error");
    }
  };

  return (
    <div>
      <div
        onClick={() => toggleExpanded(file.id)}
        className={cn(
          "grid h-[52px] cursor-pointer items-center gap-3 px-3.5 transition-colors",
          ex ? "" : "border-b border-line",
          on ? "bg-blue-50" : "hover:bg-panel-2",
        )}
        style={{ gridTemplateColumns: AUDIO_COLS }}
      >
        <button onClick={(e) => { e.stopPropagation(); toggleSelect(file.id); }} className="flex">
          <Checkbox on={on} />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); toggleExpanded(file.id); }}
          title={ex ? "Collapse" : "Show segments"}
          className={cn("flex h-[22px] w-[22px] items-center justify-center text-txt-mute transition-transform", ex && "rotate-90")}
        >
          <Icon name="chevron-down" size={15} strokeWidth={2.4} className="-rotate-90" />
        </button>
        <WaveformBars seed={index + 1} bars={20} height={22} />
        <div className="flex min-w-0 items-center gap-1.5" onClick={(event) => event.stopPropagation()}>
          {renaming ? (
            <input
              autoFocus
              value={nameDraft}
              disabled={renameAudio.isPending}
              onChange={(event) => setNameDraft(event.target.value)}
              onBlur={() => {
                if (skipNameBlur.current) {
                  skipNameBlur.current = false;
                  return;
                }
                void commitRename();
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  event.currentTarget.blur();
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  cancelRename();
                }
              }}
              className="min-w-0 flex-1 rounded border border-line bg-bg px-1.5 py-1 font-mono text-[13px] font-semibold text-txt outline-none focus:border-blue-400"
            />
          ) : (
            <>
              <span className="truncate font-mono text-[13px] font-semibold text-txt">{file.name}</span>
              <button
                title="Rename audio"
                className="flex h-5 w-5 flex-none items-center justify-center rounded text-txt-mute hover:bg-panel-3 hover:text-txt"
                onClick={() => setRenaming(true)}
              >
                <Icon name="edit" size={12} strokeWidth={2.2} />
              </button>
            </>
          )}
          {noSeg ? (
            <span className="flex flex-none text-amber-500" title="No segments — Split or Transcribe to create them">
              <Icon name="alert" size={15} strokeWidth={2.2} />
            </span>
          ) : null}
          {file.storage_kind === "external" ? (
            <span className="flex-none rounded bg-panel-2 px-1.5 py-0.5 text-[9px] font-bold uppercase text-txt-mute">metadata only</span>
          ) : null}
        </div>
        <span className="font-mono text-[12.5px] tabular-nums text-txt-dim">{fmtDur(file.duration)}</span>
        <span className={cn("justify-self-end text-[12.5px] tabular-nums", file.segments ? "font-semibold text-txt" : "text-txt-mute")}>
          {file.segments || "—"}
        </span>
        <span className="text-xs text-txt-mute">{Number.isNaN(updatedAt) ? "-" : fmtAgo(updatedAt)}</span>
        <IconButton
          icon="edit"
          title="Open segment editor"
          size={26}
          onClick={(event) => {
            event.stopPropagation();
            void navigate(`/audio/${file.id}`);
          }}
        />
      </div>
      {ex ? <InlineSegments file={file} preview={preview.data} loading={preview.isLoading} /> : null}
    </div>
  );
}
