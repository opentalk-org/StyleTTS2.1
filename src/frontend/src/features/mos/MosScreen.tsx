import { useEffect } from "react";

import { useDatasetsQuery } from "@/features/datasets/query";
import { showToast } from "@/shared/feedback/Toast";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SectionTitle } from "@/shared/ui/SectionTitle";
import { MosAudioCard } from "./MosAudioCard";
import { MosDatasetPicker } from "./MosDatasetPicker";
import { MosHistoryList } from "./MosHistoryList";
import { hasCompleteMosScores, mosRatingRequest, pairScoreDraft } from "./logic";
import { useMosPairQuery, useSaveMosRatingMutation } from "./query";
import { useMos } from "./store";

export function MosScreen() {
  const datasets = useDatasetsQuery();
  const {
    selectedDatasetIds,
    scoreA,
    scoreB,
    toggleDataset,
    setScoreA,
    setScoreB,
    resetPair,
  } = useMos();
  const pairQuery = useMosPairQuery(selectedDatasetIds);
  const saveRating = useSaveMosRatingMutation();
  const pair = pairQuery.data;
  const canSubmit = hasCompleteMosScores(scoreA, scoreB);

  useEffect(() => {
    if (!pair) return;
    resetPair(pairScoreDraft(pair.audio_a.annotations.score), pairScoreDraft(pair.audio_b.annotations.score));
  }, [pairQuery.dataUpdatedAt, pair, resetPair]);

  const submit = async (preferredAudioId: string) => {
    if (!pair) return;
    try {
      await saveRating.mutateAsync(mosRatingRequest(pair, scoreA, scoreB, preferredAudioId));
      showToast("MOS rating saved");
    } catch (error) {
      showToast("Could not save MOS rating", error instanceof Error ? error.message : undefined, "error");
    }
  };

  return (
    <div className="mx-auto max-w-[1120px] px-7 pb-20 pt-6">
      <div className="mb-5 grid gap-5 lg:grid-cols-[300px_1fr]">
        <Card className="p-4">
          <SectionTitle className="mb-2">Rating datasets</SectionTitle>
          <p className="mb-3 text-[12px] leading-relaxed text-txt-mute">
            Pick one or more datasets. Each pair comes from a single selected dataset.
          </p>
          {datasets.isLoading ? (
            <div className="text-[13px] text-txt-mute">Loading datasets…</div>
          ) : datasets.isError ? (
            <div className="text-[13px] text-red-600">Could not load datasets.</div>
          ) : (
            <MosDatasetPicker datasets={datasets.data ?? []} selectedIds={selectedDatasetIds} onToggle={toggleDataset} />
          )}
        </Card>

        <div className="min-w-0">
          {!selectedDatasetIds.length ? (
            <Card><EmptyState icon="gauge" title="Choose datasets to start rating" description="MOS presents two random playable audio files at a time." /></Card>
          ) : pairQuery.isLoading ? (
            <Card className="p-8 text-sm text-txt-mute">Selecting a random pair…</Card>
          ) : pairQuery.isError || !pair ? (
            <Card>
              <EmptyState
                icon="alert"
                title="No pair is available"
                description="A selected dataset must contain at least two non-virtual audio files."
                action={<Button icon="refresh" onClick={() => pairQuery.refetch()}>Try again</Button>}
              />
            </Card>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="grid gap-4 xl:grid-cols-2">
                <MosAudioCard
                  label="A"
                  audio={pair.audio_a}
                  score={scoreA}
                  disabled={saveRating.isPending || !canSubmit}
                  onScore={setScoreA}
                  onChoose={() => void submit(pair.audio_a.id)}
                />
                <MosAudioCard
                  label="B"
                  audio={pair.audio_b}
                  score={scoreB}
                  disabled={saveRating.isPending || !canSubmit}
                  onScore={setScoreB}
                  onChoose={() => void submit(pair.audio_b.id)}
                />
              </div>
              <div className="rounded-lg border border-line bg-panel px-4 py-3 text-[12px] text-txt-mute">Enter both scores, then choose the better sample to save and load the next pair.</div>
            </div>
          )}
        </div>
      </div>
      {selectedDatasetIds.length ? <MosHistoryList datasetIds={selectedDatasetIds} /> : null}
    </div>
  );
}
