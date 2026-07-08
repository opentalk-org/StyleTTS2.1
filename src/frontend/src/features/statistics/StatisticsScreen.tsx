import { useEffect, useState } from "react";

import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { fmtAgo } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import { EmptyState } from "@/shared/ui/EmptyState";
import { SectionTitle } from "@/shared/ui/SectionTitle";
import { Select } from "@/shared/ui/Select";
import { Tabs } from "@/shared/ui/Tabs";

import type { StatisticsPayload } from "./api";
import { ChartCard } from "./charts/ChartCard";
import { HBars } from "./charts/HBars";
import { Histogram } from "./charts/Histogram";
import { RankList } from "./charts/RankList";
import { StatTile } from "./charts/StatTile";
import { audioHistograms, audioTiles, corpusData, speakerDuration, textWarnings, type CorpusTab } from "./logic";
import { useStatisticsActions, useStatisticsEntriesQuery, useStatisticsEntryQuery } from "./query";

export function StatisticsScreen() {
  const entriesQuery = useStatisticsEntriesQuery();
  const entries = entriesQuery.data ?? [];
  const [entryId, setEntryId] = useState<string | null>(null);
  const [tab, setTab] = useState<CorpusTab>("transcript");
  const actions = useStatisticsActions();

  useEffect(() => {
    if (entries.length === 0) {
      if (entryId !== null) setEntryId(null);
      return;
    }
    if (entryId === null || !entries.some((e) => e.id === entryId)) {
      setEntryId(entries[0]!.id);
    }
  }, [entries, entryId]);

  const entryQuery = useStatisticsEntryQuery(entryId);
  const summary = entries.find((e) => e.id === entryId) ?? null;

  const remove = () => {
    if (!summary) return;
    askConfirm({
      title: "Delete statistics entry?",
      desc: `Delete “${summary.name}”. You can recompute it anytime.`,
      danger: true,
      label: "Delete entry",
      onConfirm: () => {
        void actions.remove(summary.id).then(() => setEntryId(null));
      },
    });
  };

  if (entriesQuery.isLoading) {
    return <div className="px-7 pt-10 text-[13px] text-txt-mute">Loading statistics…</div>;
  }

  if (entries.length === 0) {
    return (
      <div className="mx-auto max-w-[1180px] px-7 pt-10">
        <EmptyState
          icon="bar-chart"
          title="No statistics yet"
          description="Run a dataset statistics workflow (AudioSource → LoadAudio → LoadAudioSegments → AnalyzeAudioFeatures → AggregateDatasetStatistics → SaveStatisticsEntry) to compute and save an entry."
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1180px] px-7 pt-5 pb-[70px]">
      <div className="mb-[18px] flex flex-wrap items-center gap-3">
        <div className="min-w-[280px]">
          <Select
            variant="mini"
            value={entryId ?? ""}
            onChange={setEntryId}
            options={entries.map((e) => ({ value: e.id, label: `${e.name}  ·  ${e.file_count} files` }))}
          />
        </div>
        {summary ? (
          <span className="text-[12px] text-txt-mute">Computed {fmtAgo(new Date(summary.created_at).getTime())}</span>
        ) : null}
        <div className="flex-1" />
        <IconButton icon="trash" title="Delete" onClick={remove} />
      </div>

      {entryQuery.isLoading || !entryQuery.data ? (
        <div className="px-1 py-16 text-center text-[13px] text-txt-mute">Loading entry…</div>
      ) : (
        <StatisticsBody payload={entryQuery.data.payload} tab={tab} onTab={setTab} />
      )}
    </div>
  );
}

function StatisticsBody({ payload, tab, onTab }: { payload: StatisticsPayload; tab: CorpusTab; onTab: (t: CorpusTab) => void }) {
  const histograms = audioHistograms(payload);
  const tiles = audioTiles(payload);
  const warnings = textWarnings(payload);
  const speakers = speakerDuration(payload);
  const corpus = corpusData(payload, tab);

  return (
    <>
      {warnings.length > 0 ? (
        <div className="mb-[22px] rounded-[10px] border border-amber-500/40 bg-amber-50 px-4 py-[14px]">
          <div className="mb-2.5 flex items-center gap-2">
            <Icon name="alert" size={16} strokeWidth={2.2} className="text-amber-600" />
            <span className="text-[13px] font-bold text-amber-700">
              {warnings.length} file{warnings.length === 1 ? "" : "s"} outside the trainable text-length range
            </span>
          </div>
          <div className="flex flex-col gap-1.5">
            {warnings.slice(0, 12).map((w, i) => (
              <div key={i} className="flex items-center gap-2.5 text-[12.5px]">
                <span className="min-w-[180px] truncate font-mono font-semibold text-txt">{w.file}</span>
                <span className="min-w-[170px] font-semibold text-amber-700">{w.why}</span>
                <span className="tabular-nums text-txt-mute">{w.detail}</span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <SectionTitle className="my-2 mb-[14px] tracking-[0.06em]">Audio · {payload.file_count} files</SectionTitle>
      <div className="mb-[14px] grid grid-cols-3 gap-[14px]">
        {histograms.map((h) => (
          <ChartCard key={h.title} title={h.title} unit={h.unit}>
            <Histogram bins={h.bins} ticks={h.ticks} tone={h.tone} />
          </ChartCard>
        ))}
      </div>
      <div className="mb-[14px] grid grid-cols-4 gap-[14px]">
        {tiles.map((t) => (
          <StatTile key={t.label} label={t.label} value={t.value} sub={t.sub} tone={t.tone} />
        ))}
      </div>
      {speakers.length > 0 ? (
        <ChartCard title="Duration per speaker" unit="time" span>
          <HBars items={speakers} tone="blue" />
        </ChartCard>
      ) : null}

      <div className="mt-[26px] mb-[14px] flex items-center gap-[14px]">
        <SectionTitle className="tracking-[0.06em]">Text corpus</SectionTitle>
        <Tabs
          value={tab}
          onChange={(v) => onTab(v as CorpusTab)}
          options={[
            { value: "transcript", label: "Transcript" },
            { value: "ipa", label: "IPA" },
          ]}
        />
      </div>

      {corpus.available ? (
        <>
          <div className="mb-[14px] grid grid-cols-2 gap-[14px]">
            <ChartCard title="Length per file" unit={corpus.unit}>
              <Histogram bins={corpus.lengthBins} ticks={corpus.lengthTicks} tone="blue" />
            </ChartCard>
            <ChartCard title="Total length per speaker" unit={corpus.unit}>
              <HBars items={corpus.speakerLength} tone="emerald" />
            </ChartCard>
          </div>
          <ChartCard title={`1-gram frequency · ${corpus.unit}`} unit="count" span>
            <HBars items={corpus.grams1} tone="blue" />
          </ChartCard>
          <div className="mt-[14px] grid grid-cols-2 gap-[14px]">
            <ChartCard title="Top 10 trigrams" unit="count">
              <RankList items={corpus.trigramsTop} tone="blue" />
            </ChartCard>
            <ChartCard title="Bottom 10 trigrams" unit="count">
              <RankList items={corpus.trigramsBottom} tone="amber" />
            </ChartCard>
          </div>
        </>
      ) : (
        <EmptyState
          icon="alert"
          title="No phoneme data"
          description="These segments were not phonemized, so IPA statistics are not available. Run phonemization before computing statistics to populate this tab."
        />
      )}
    </>
  );
}
