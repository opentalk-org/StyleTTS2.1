import { useEffect } from "react";

import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { Icon, type IconName } from "@/shared/icons";
import { IconButton } from "@/shared/ui/IconButton";
import { EmptyState } from "@/shared/ui/EmptyState";
import { cn } from "@/shared/ui/cn";

import type { StatisticsPayload } from "./api";
import { ComputeStatistics } from "./ComputeStatistics";
import { StatisticsEntryPicker } from "./StatisticsEntryPicker";
import { GRID_COLS, StatSection } from "./StatSection";
import { ChartCard } from "./charts/ChartCard";
import { HBars } from "./charts/HBars";
import { Heatmap } from "./charts/Heatmap";
import { Histogram } from "./charts/Histogram";
import { RankList } from "./charts/RankList";
import { Scatter } from "./charts/Scatter";
import { StatTile } from "./charts/StatTile";
import { audioHistogramGroups, audioTiles, corpusData, rateScatters, textWarnings, type CorpusTab, type HistogramConfig } from "./logic";
import { useStatisticsActions, useStatisticsEntriesQuery, useStatisticsEntryQuery } from "./query";
import { reconcileStatisticsEntryId } from "./selection";
import { useStatisticsUi } from "./store";

export function StatisticsScreen() {
  const entriesQuery = useStatisticsEntriesQuery();
  const entries = entriesQuery.data ?? [];
  const entryId = useStatisticsUi((s) => s.entryId);
  const setEntryId = useStatisticsUi((s) => s.setEntryId);
  const tab = useStatisticsUi((s) => s.tab);
  const setTab = useStatisticsUi((s) => s.setTab);
  const actions = useStatisticsActions();

  useEffect(() => {
    const reconciled = reconcileStatisticsEntryId(entryId, entriesQuery.data);
    if (reconciled !== entryId) setEntryId(reconciled);
  }, [entriesQuery.data, entryId]);

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
        <div className="mb-5 flex justify-center">
          <ComputeStatistics />
        </div>
        <EmptyState
          icon="bar-chart"
          title="No statistics yet"
          description="Pick a dataset above, choose database-only or acoustic analysis, then compute all files or a random sample."
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1180px] px-7 pt-5 pb-[70px]">
      {/* Sticky toolbar keeps the entry picker and compute controls in reach while scrolling
          through a long report. */}
      <div className="sticky top-0 z-30 -mx-7 mb-6 border-b border-line bg-app/85 px-7 py-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3">
          <StatisticsEntryPicker entries={entries} value={entryId} onChange={setEntryId} />
          <div className="flex-1" />
          <ComputeStatistics />
          <IconButton
            icon="trash"
            title="Delete"
            disabled={!summary}
            onClick={remove}
            className="disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-txt-mute"
          />
        </div>
      </div>

      {entryId === null ? (
        <EmptyState
          icon="bar-chart"
          title="Select a statistics entry"
          description="Choose a saved report from the picker above to view its statistics."
        />
      ) : entryQuery.isLoading || !entryQuery.data ? (
        <div className="px-1 py-16 text-center text-[13px] text-txt-mute">Loading entry…</div>
      ) : (
        <StatisticsBody payload={entryQuery.data.payload} tab={tab} onTab={setTab} />
      )}
    </div>
  );
}

const CORPUS_MODES: { value: CorpusTab; label: string; caption: string }[] = [
  { value: "transcript", label: "Transcript", caption: "Raw written characters" },
  { value: "ipa", label: "IPA", caption: "Phonemes, after phonemization" },
];

function CorpusModeSwitch({ value, onChange }: { value: CorpusTab; onChange: (t: CorpusTab) => void }) {
  return (
    <div className="grid grid-cols-2 gap-2 rounded-xl border border-line bg-panel-2 p-1.5">
      {CORPUS_MODES.map((m) => {
        const on = value === m.value;
        return (
          <button
            key={m.value}
            onClick={() => onChange(m.value)}
            className={cn(
              "flex flex-col items-start gap-0.5 rounded-lg px-4 py-2.5 text-left transition-colors cursor-pointer",
              on ? "bg-panel shadow-sm ring-1 ring-blue-500/30" : "hover:bg-panel/50",
            )}
          >
            <span className={cn("text-[13.5px] font-bold", on ? "text-blue-600" : "text-txt-dim")}>{m.label}</span>
            <span className="text-[11px] text-txt-mute">{m.caption}</span>
          </button>
        );
      })}
    </div>
  );
}

function HistogramChart({ h }: { h: HistogramConfig }) {
  return (
    <ChartCard title={h.title} unit={h.unit}>
      <Histogram edges={h.edges} counts={h.counts} underflow={h.underflow} overflow={h.overflow} tone={h.tone} countLabel={h.countLabel ?? "files"} />
    </ChartCard>
  );
}

function MetaChip({ icon, label, value, tone = "default" }: { icon: IconName; label: string; value: string; tone?: "default" | "amber" }) {
  const accent = tone === "amber" ? "text-amber-600" : "text-txt-mute";
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-[7px] text-[11.5px]">
      <Icon name={icon} size={13} strokeWidth={2.2} className={accent} />
      <span className="text-txt-mute">{label}</span>
      <span className="font-semibold text-txt">{value}</span>
    </span>
  );
}

function StatisticsBody({ payload, tab, onTab }: { payload: StatisticsPayload; tab: CorpusTab; onTab: (t: CorpusTab) => void }) {
  const histogramGroups = audioHistogramGroups(payload);
  const tiles = audioTiles(payload);
  const warnings = textWarnings(payload);
  const corpus = corpusData(payload, tab);
  const scatters = rateScatters(payload, tab);

  const scope =
    payload.sample_scope.selection === "random" ? `Random sample · ${payload.sample_scope.actual_count.toLocaleString()}` : "All files";

  return (
    <div className="space-y-10">
      {warnings.length > 0 ? (
        <div className="rounded-[10px] border border-amber-500/40 bg-amber-50 px-4 py-[14px]">
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

      {/* Overview — the headline numbers, up top where a reader looks first. */}
      <StatSection title="Overview" caption="Dataset totals and the scope this report was computed over.">
        <div className="mb-[14px] flex flex-wrap gap-2">
          <MetaChip icon="folder-open" label="Files" value={payload.file_count.toLocaleString()} />
          <MetaChip
            icon={payload.computation_mode === "database" ? "database" : "activity"}
            label="Mode"
            value={payload.computation_mode === "database" ? "Database only" : "With audio"}
          />
          <MetaChip icon="sliders" label="Scope" value={scope} tone={payload.sample_scope.selection === "random" ? "amber" : "default"} />
        </div>
        <div className={`grid gap-[14px] ${GRID_COLS[tiles.length] ?? "grid-cols-4"}`}>
          {tiles.map((t) => (
            <StatTile key={t.label} label={t.label} value={t.value} sub={t.sub} tone={t.tone} />
          ))}
        </div>
      </StatSection>

      {/* Audio distributions, one clearly-labeled section per concern. */}
      {histogramGroups.map((group) => (
        <StatSection key={group.key} title={group.title} caption={group.caption}>
          <div className={`grid gap-[14px] ${GRID_COLS[Math.min(group.items.length, 3)]}`}>
            {group.items.map((h) => (
              <HistogramChart key={h.title} h={h} />
            ))}
          </div>
        </StatSection>
      ))}

      {/* Text corpus — a distinct chapter set apart with a rule, led by a full-width mode
          switch that visibly governs every chart below it. */}
      <div className="border-t border-line pt-9">
        <h2 className="text-[16px] font-extrabold tracking-tight text-txt">Text corpus</h2>
        <p className="mt-1 text-[12px] leading-snug text-txt-mute">
          Token distribution and structure of the corpus — pick which representation to analyze.
        </p>
        <div className="mt-4">
          <CorpusModeSwitch value={tab} onChange={onTab} />
        </div>

        {corpus.available ? (
          <div className="mt-8 space-y-8">
            <StatSection title="Length & structure" caption="How long segments are and how densely they are punctuated.">
              <div className="grid grid-cols-3 gap-[14px]">
                <ChartCard title="Length per file" unit={corpus.unit}>
                  <Histogram edges={corpus.lengthEdges} counts={corpus.lengthCounts} underflow={corpus.lengthUnderflow} overflow={corpus.lengthOverflow} tone="blue" />
                </ChartCard>
                <ChartCard title={'Sentence markers ". " or final "."'} unit="per file">
                  <Histogram edges={corpus.sentenceMarkerEdges} counts={corpus.sentenceMarkerCounts} underflow={corpus.sentenceMarkerUnderflow} overflow={corpus.sentenceMarkerOverflow} tone="blue" />
                </ChartCard>
                <ChartCard title={'Comma markers ", " or final ","'} unit="per file">
                  <Histogram edges={corpus.commaMarkerEdges} counts={corpus.commaMarkerCounts} underflow={corpus.commaMarkerUnderflow} overflow={corpus.commaMarkerOverflow} tone="emerald" />
                </ChartCard>
              </div>
            </StatSection>

            <StatSection title="Speaking rate" caption="Per-sample rate estimates versus clip duration and total count.">
              <div className="grid grid-cols-2 gap-[14px]">
                {scatters.map((s) => (
                  <ChartCard key={`${s.title} · ${s.unit}`} title={s.title} unit={s.unit}>
                    {s.points.length ? (
                      <Scatter points={s.points} xLabel={s.xLabel} yLabel={s.yLabel} tone={s.tone} />
                    ) : (
                      <div className="py-16 text-center text-[12px] text-txt-mute">No sample-level timing available.</div>
                    )}
                  </ChartCard>
                ))}
              </div>
            </StatSection>

            <StatSection title="Per-speaker volume" caption={`Total ${corpus.unit} contributed by each speaker.`}>
              <ChartCard title="Total length per speaker" unit={corpus.unit}>
                <HBars items={corpus.speakerLength} tone="emerald" />
              </ChartCard>
            </StatSection>

            <StatSection title="Token frequency" caption={`Most and least common ${corpus.unit === "phonemes" ? "phoneme" : "character"} sequences.`}>
              <div className="space-y-[14px]">
                <ChartCard title={`1-gram frequency · ${corpus.unit}`} unit="count">
                  <HBars items={corpus.grams1} tone="blue" />
                </ChartCard>
                {corpus.bigramMatrix.labels.length > 0 ? (
                  <ChartCard title={`2-gram frequency · ${corpus.unit}`} unit="count">
                    <Heatmap data={corpus.bigramMatrix} unit={corpus.unit === "phonemes" ? "phoneme" : "character"} />
                  </ChartCard>
                ) : null}
                <div className="grid grid-cols-2 gap-[14px]">
                  <ChartCard title="Top 10 trigrams" unit="count">
                    <RankList items={corpus.trigramsTop} tone="blue" />
                  </ChartCard>
                  <ChartCard title="Bottom 10 trigrams" unit="count">
                    <RankList items={corpus.trigramsBottom} tone="amber" />
                  </ChartCard>
                </div>
              </div>
            </StatSection>
          </div>
        ) : (
          <div className="mt-8">
            <EmptyState
              icon="alert"
              title="No phoneme data"
              description="These segments were not phonemized, so IPA statistics are not available. Run phonemization before computing statistics to populate this tab."
            />
          </div>
        )}
      </div>
    </div>
  );
}
