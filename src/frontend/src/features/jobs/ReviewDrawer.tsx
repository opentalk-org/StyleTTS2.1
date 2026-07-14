import { useEffect, useState } from "react";

import { VirtualTable } from "@/shared/data/VirtualTable";
import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { Modal } from "@/shared/feedback/Modal";
import { WaveformPlayer } from "@/shared/media/WaveformPlayer";
import { Badge } from "@/shared/ui/Badge";
import { Button } from "@/shared/ui/Button";
import { Card } from "@/shared/ui/Card";
import { cn } from "@/shared/ui/cn";
import type { ReviewGroup, ReviewItem } from "./api";
import { useReviewDecision, useReviewQuery, useReviewsQuery } from "./query";
import { reviewMediaUrl, reviewStateTone, reviewTone } from "./reviews";

export function ReviewDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const summaries = useReviewsQuery(runId);
  const [reviewId, setReviewId] = useState<string | null>(null);
  const [groupKey, setGroupKey] = useState<string | null>(null);
  const review = useReviewQuery(reviewId);
  const decision = useReviewDecision();

  useEffect(() => {
    const first = summaries.data?.[0];
    if (first && reviewId === null) setReviewId(first.id);
  }, [reviewId, summaries.data]);

  useEffect(() => {
    const first = review.data?.payload.groups[0];
    if (first) setGroupKey(first.key);
  }, [review.data?.id]);

  const data = review.data;
  const group = data?.payload.groups.find((candidate) => candidate.key === groupKey) ?? data?.payload.groups[0];
  const submitDecision = (next: "approved" | "rejected") => {
    if (!data) return;
    askConfirm({
      title: next === "approved" ? "Approve this review?" : "Reject this review?",
      desc: next === "approved"
        ? "Approval starts the linked continuation workflow once."
        : "Rejection records the decision without starting the continuation.",
      danger: next === "rejected",
      label: next === "approved" ? "Approve" : "Reject",
      onConfirm: () => decision.mutate({ reviewId: data.id, decision: next }),
    });
  };

  return (
    <Modal
      icon="check-circle"
      title={data?.title ?? "Workflow review"}
      desc={`Produced by ${runId}`}
      onClose={onClose}
      maxWidth={980}
      footer={data?.state === "pending" ? (
        <>
          <div className="flex-1" />
          <Button variant="danger" onClick={() => submitDecision("rejected")} disabled={decision.isPending}>Reject</Button>
          <Button variant="primary" icon="check" onClick={() => submitDecision("approved")} disabled={decision.isPending}>Approve</Button>
        </>
      ) : null}
    >
      {summaries.isLoading || review.isLoading ? <div className="py-8 text-sm text-txt-mute">Loading review...</div> : null}
      {summaries.isError || review.isError ? <div className="py-8 text-sm text-red-600">Could not load this review.</div> : null}
      {data ? (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={reviewStateTone(data.state)}>{data.state}</Badge>
            <Badge>{data.kind}</Badge>
            {data.continuation_run_id ? <span className="font-mono text-[11px] text-txt-mute">Continuation: {data.continuation_run_id}</span> : null}
          </div>
          {(summaries.data?.length ?? 0) > 1 ? (
            <div className="flex flex-wrap gap-2">
              {summaries.data!.map((summary) => (
                <Button key={summary.id} size="sm" variant={summary.id === data.id ? "primary" : "secondary"} onClick={() => setReviewId(summary.id)}>
                  {summary.title}
                </Button>
              ))}
            </div>
          ) : null}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {data.payload.metrics.map((metric) => (
              <Card key={metric.key} className="p-3">
                <Badge tone={reviewTone(metric.tone)}>{metric.label}</Badge>
                <div className="mt-2 text-xl font-bold tabular-nums text-txt">{metric.value}</div>
              </Card>
            ))}
          </div>
          {data.payload.warnings.length ? (
            <Card className="border-amber-200 bg-amber-50 p-3 text-[13px] text-amber-800">
              {data.payload.warnings.map((warning) => <div key={warning}>{warning}</div>)}
            </Card>
          ) : null}
          {data.payload.groups.length ? (
            <>
              <div className="flex flex-wrap gap-2">
                {data.payload.groups.map((candidate) => (
                  <button
                    key={candidate.key}
                    onClick={() => setGroupKey(candidate.key)}
                    className={cn("rounded-md border px-3 py-2 text-left text-xs", candidate.key === group?.key ? "border-blue-500 bg-blue-50 text-blue-700" : "border-line bg-panel text-txt-dim")}
                  >
                    <span className="font-semibold">{candidate.title}</span> · {candidate.items.length.toLocaleString()}
                  </button>
                ))}
              </div>
              {group ? <ReviewGroupList group={group} /> : null}
            </>
          ) : <div className="text-sm text-txt-mute">This review has no item groups.</div>}
        </div>
      ) : null}
    </Modal>
  );
}

function ReviewGroupList({ group }: { group: ReviewGroup }) {
  return (
    <div>
      <div className="mb-2 text-xs text-txt-mute">{group.explanation}</div>
      {group.items.length ? (
        <VirtualTable
          count={group.items.length}
          estimateRowHeight={150}
          className="h-[48vh] rounded-lg border border-line"
          renderRow={(index) => <ReviewItemRow item={group.items[index]!} index={index} />}
        />
      ) : <div className="rounded-lg border border-dashed border-line p-8 text-center text-sm text-txt-mute">No items in this group.</div>}
    </div>
  );
}

function ReviewItemRow({ item, index }: { item: ReviewItem; index: number }) {
  return (
    <div className="space-y-2 border-b border-line p-3 last:border-b-0">
      <div>
        <div className="text-sm font-semibold text-txt">{item.title}</div>
        {item.subtitle ? <div className="text-xs text-txt-mute">{item.subtitle}</div> : null}
      </div>
      {item.fields.length ? (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {item.fields.map((field) => <span key={field.key}><span className="text-txt-mute">{field.label}:</span> {field.value}</span>)}
        </div>
      ) : null}
      {item.media.map((media) => (
        <WaveformPlayer
          key={`${media.audio_file_id}:${media.segment_id}`}
          seed={index + media.segment_id.length}
          duration={media.duration_seconds}
          fileName={media.name}
          src={reviewMediaUrl(media)}
          clipStart={media.start_seconds}
          clipEnd={media.end_seconds}
        />
      ))}
    </div>
  );
}
