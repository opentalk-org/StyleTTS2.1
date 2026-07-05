import { useState } from "react";

import { askConfirm } from "@/shared/feedback/ConfirmDialog";
import { showToast } from "@/shared/feedback/Toast";
import { fmtAgo } from "@/shared/format";
import { Icon } from "@/shared/icons";
import { Button } from "@/shared/ui/Button";
import { IconButton } from "@/shared/ui/IconButton";
import { SectionTitle } from "@/shared/ui/SectionTitle";
import { Select } from "@/shared/ui/Select";
import { Tabs } from "@/shared/ui/Tabs";
import { seedStatEntries } from "@/mock/data";
import type { StatEntry } from "@/mock/types";

import { ChartCard } from "./charts/ChartCard";
import { HBars } from "./charts/HBars";
import { Histogram } from "./charts/Histogram";
import { RankList } from "./charts/RankList";
import { StatTile } from "./charts/StatTile";
import { AUDIO_HISTOGRAMS, SPEAKER_DURATION, TEXT_WARNINGS, corpusData, type CorpusTab } from "./logic";

export function StatisticsScreen() {
  const [entries, setEntries] = useState<StatEntry[]>(seedStatEntries);
  const [entryId, setEntryId] = useState(entries[0]!.id);
  const [tab, setTab] = useState<CorpusTab>("transcript");

  const entry = entries.find((e) => e.id === entryId) ?? entries[0];
  if (!entry) return null;
  const corpus = corpusData(tab);

  const remove = () =>
    askConfirm({
      title: "Delete statistics entry?",
      desc: `Delete “${entry.id}”. You can recompute it anytime.`,
      danger: true,
      label: "Delete entry",
      onConfirm: () => {
        const rest = entries.filter((e) => e.id !== entry.id);
        setEntries(rest);
        setEntryId(rest[0]?.id ?? "");
        showToast("Entry deleted", undefined, "error");
      },
    });

  return (
    <div className="mx-auto max-w-[1180px] px-7 pt-5 pb-[70px]">
      <div className="mb-[18px] flex flex-wrap items-center gap-3">
        <div className="min-w-[240px]">
          <Select
            variant="mini"
            value={entryId}
            onChange={setEntryId}
            options={entries.map((e) => ({ value: e.id, label: `${e.id}  ·  ${e.files} files` }))}
          />
        </div>
        <span className="text-[12px] text-txt-mute">Computed {fmtAgo(entry.created)}</span>
        <div className="flex-1" />
        <Button icon="refresh" onClick={() => showToast("Statistics refreshed", entry.id)}>
          Refresh
        </Button>
        <IconButton icon="trash" title="Delete" onClick={remove} />
      </div>

      <div className="mb-[22px] rounded-[10px] border border-amber-500/40 bg-amber-50 px-4 py-[14px]">
        <div className="mb-2.5 flex items-center gap-2">
          <Icon name="alert" size={16} strokeWidth={2.2} className="text-amber-600" />
          <span className="text-[13px] font-bold text-amber-700">
            {TEXT_WARNINGS.length} files outside the trainable text-length range
          </span>
        </div>
        <div className="flex flex-col gap-1.5">
          {TEXT_WARNINGS.map((w, i) => (
            <div key={i} className="flex items-center gap-2.5 text-[12.5px]">
              <span className="min-w-[130px] font-mono font-semibold text-txt">{w.file}</span>
              <span className="min-w-[170px] font-semibold text-amber-700">{w.why}</span>
              <span className="tabular-nums text-txt-mute">{w.detail}</span>
            </div>
          ))}
        </div>
      </div>

      <SectionTitle className="my-2 mb-[14px] tracking-[0.06em]">Audio · {entry.files} files</SectionTitle>
      <div className="mb-[14px] grid grid-cols-3 gap-[14px]">
        {AUDIO_HISTOGRAMS.map((h) => (
          <ChartCard key={h.title} title={h.title} unit={h.unit}>
            <Histogram bins={h.bins} xmin={h.xmin} xmid={h.xmid} xmax={h.xmax} tone={h.tone} />
          </ChartCard>
        ))}
      </div>
      <div className="mb-[14px] grid grid-cols-4 gap-[14px]">
        <StatTile label="Clipped files" value="3" sub={`of ${entry.files} (0.6%)`} tone="red" />
        <StatTile label="Max peak" value="0.998" sub="dBFS −0.02" tone="amber" />
        <StatTile label="Files > −1 dBFS" value="12" sub="review recommended" tone="amber" />
        <StatTile label="Total duration" value="12.2 h" sub="avg 0:31 / file" tone="blue" />
      </div>
      <ChartCard title="Duration per speaker" unit="hours" span>
        <HBars items={SPEAKER_DURATION} tone="blue" />
      </ChartCard>

      <div className="mt-[26px] mb-[14px] flex items-center gap-[14px]">
        <SectionTitle className="tracking-[0.06em]">Text corpus</SectionTitle>
        <Tabs
          value={tab}
          onChange={(v) => setTab(v as CorpusTab)}
          options={[
            { value: "transcript", label: "Transcript" },
            { value: "ipa", label: "IPA" },
          ]}
        />
      </div>
      <div className="mb-[14px] grid grid-cols-2 gap-[14px]">
        <ChartCard title="Length per file" unit={corpus.unit}>
          <Histogram
            bins={corpus.lengthBins}
            xmin={corpus.lengthAxis.xmin}
            xmid={corpus.lengthAxis.xmid}
            xmax={corpus.lengthAxis.xmax}
            tone="blue"
          />
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
    </div>
  );
}
