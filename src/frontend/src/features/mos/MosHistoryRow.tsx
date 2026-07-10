import { useState } from "react";

import { AudioScoreInput, formatAudioScore } from "@/features/audio/AudioScoreInput";
import { showToast } from "@/shared/feedback/Toast";
import { fmtAgo } from "@/shared/format";
import { Button } from "@/shared/ui/Button";
import type { MosHistoryItem, MosRatingUpdateRequest } from "./api";
import { hasCompleteMosScores, mosRatingUpdateRequest } from "./logic";

export function MosHistoryRow({
  item,
  pending,
  onUpdate,
  onUndo,
}: {
  item: MosHistoryItem;
  pending: boolean;
  onUpdate: (id: string, payload: MosRatingUpdateRequest) => Promise<void>;
  onUndo: (id: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [scoreA, setScoreA] = useState(formatAudioScore(item.score_a));
  const [scoreB, setScoreB] = useState(formatAudioScore(item.score_b));
  const canSave = hasCompleteMosScores(scoreA, scoreB);
  const preferred = item.preferred_audio_id === item.audio_a_id ? "A" : "B";

  const update = async (preferredAudioId: string) => {
    try {
      await onUpdate(item.id, mosRatingUpdateRequest(scoreA, scoreB, preferredAudioId));
      setEditing(false);
      showToast("MOS comparison changed");
    } catch (error) {
      showToast("Could not change MOS comparison", error instanceof Error ? error.message : undefined, "error");
    }
  };

  const undo = async () => {
    if (!window.confirm("Undo the newest MOS comparison and restore its previous scores?")) return;
    try {
      await onUndo(item.id);
      showToast("MOS comparison undone");
    } catch (error) {
      showToast("Could not undo MOS comparison", error instanceof Error ? error.message : undefined, "error");
    }
  };

  return (
    <div className="rounded-lg border border-line bg-panel px-3.5 py-3">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[12px]">
            <span className="font-bold text-txt">A · {item.audio_a.name}</span>
            <span className="font-mono font-semibold text-blue-600">{item.score_a.toFixed(3)}</span>
          </div>
          <div className="flex items-center gap-2 text-[12px]">
            <span className="font-bold text-txt">B · {item.audio_b.name}</span>
            <span className="font-mono font-semibold text-blue-600">{item.score_b.toFixed(3)}</span>
          </div>
        </div>
        <div className="rounded-full bg-blue-50 px-2.5 py-1 text-[11px] font-bold text-blue-700">Better: {preferred}</div>
        <div className="w-16 text-right text-[11px] text-txt-mute">{fmtAgo(Date.parse(item.created_at))}</div>
        {item.can_modify && !editing ? <Button size="sm" icon="edit" onClick={() => setEditing(true)}>Change</Button> : null}
        {item.can_modify ? <Button size="sm" variant="danger" icon="rewind" disabled={pending} onClick={() => void undo()}>Undo</Button> : null}
      </div>
      {editing ? (
        <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
          <AudioScoreInput value={scoreA} disabled={pending} label="A score" onChange={setScoreA} onCancel={() => setScoreA(formatAudioScore(item.score_a))} />
          <AudioScoreInput value={scoreB} disabled={pending} label="B score" onChange={setScoreB} onCancel={() => setScoreB(formatAudioScore(item.score_b))} />
          <div className="flex-1" />
          <Button size="sm" disabled={pending} onClick={() => setEditing(false)}>Cancel</Button>
          <Button size="sm" variant="primary" disabled={pending || !canSave} onClick={() => void update(item.audio_a_id)}>Choose A and save</Button>
          <Button size="sm" variant="primary" disabled={pending || !canSave} onClick={() => void update(item.audio_b_id)}>Choose B and save</Button>
        </div>
      ) : null}
    </div>
  );
}
