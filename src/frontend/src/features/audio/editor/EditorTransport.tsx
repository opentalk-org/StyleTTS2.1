import { fmtClock } from "@/shared/format";
import { Icon, type IconName } from "@/shared/icons";
import { Slider } from "@/shared/ui/form/Slider";
import { cn } from "@/shared/ui/cn";
import { SegmentTimeline } from "../SegmentTimeline";
import { useEditor } from "../editorStore";

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];

function TransportButton({ icon, title, flip, onClick }: { icon: IconName; title: string; flip?: boolean; onClick: () => void }) {
  return (
    <button title={title} onClick={onClick} className={cn("flex h-9 w-9 items-center justify-center rounded-md bg-panel-2 text-txt hover:bg-panel-3", flip && "-scale-x-100")}>
      <Icon name={icon} size={16} strokeWidth={2.2} />
    </button>
  );
}

export function EditorTransport({
  waveformPending,
  minimapPeaks,
  viewPeaks,
  seed,
  onDownload,
}: {
  waveformPending: boolean;
  minimapPeaks?: [number, number][];
  viewPeaks?: [number, number][];
  seed: number;
  onDownload: () => void;
}) {
  const {
    segs, dur, playPos, playing, speed, volume, loop, viewStart, viewEnd, segSel,
    seek, select, setView, setSegTime, togglePlay, setSpeed, setVolume, toggleLoop, zoomOut, zoomIn,
  } = useEditor();
  const selectedSegment = segs.find((segment) => segment.id === segSel);
  return (
    <div className="mb-4 rounded-[10px] border border-line bg-panel p-4">
      {waveformPending ? <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold text-txt-mute"><Icon name="loader" size={12} className="animate-spin text-blue-500" />Generating waveform…</div> : null}
      <SegmentTimeline
        segs={segs}
        dur={dur}
        playPos={playPos}
        selId={segSel}
        viewStart={viewStart}
        viewEnd={viewEnd}
        seed={seed}
        onSeek={seek}
        onSelect={select}
        onSetView={setView}
        onSegTime={setSegTime}
        minimapPeaks={minimapPeaks}
        viewPeaks={viewPeaks}
      />
      <div className="mt-3.5 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <TransportButton icon="skip-back" title="To start" onClick={() => seek(0)} />
          <TransportButton icon="rewind" title="Back 1s" onClick={() => seek(playPos - 1)} />
          <button onClick={togglePlay} title="Play / pause" className="flex h-[46px] w-[46px] items-center justify-center rounded-full bg-blue-500 text-white hover:bg-blue-600">
            <Icon name={playing ? "pause" : "play"} size={20} strokeWidth={2.2} />
          </button>
          <TransportButton icon="rewind" title="Forward 1s" flip onClick={() => seek(playPos + 1)} />
          <TransportButton icon="skip-fwd" title="To end" onClick={() => seek(dur)} />
        </div>
        <div className="min-w-[150px] font-mono text-sm font-semibold tabular-nums text-txt">{fmtClock(playPos)}<span className="text-txt-mute"> / {fmtClock(dur)}</span></div>
        <div className="flex items-center gap-1.5">
          <Icon name="gauge" size={15} strokeWidth={2} className="text-txt-mute" />
          <div className="relative">
            <select value={String(speed)} onChange={(event) => setSpeed(parseFloat(event.target.value))} className="h-8 appearance-none rounded-md bg-panel-2 pl-2.5 pr-6 text-[12.5px] font-semibold text-txt outline-none">
              {SPEEDS.map((value) => <option key={value} value={value}>{value}×</option>)}
            </select>
            <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-txt-dim"><Icon name="chevron-down" size={12} strokeWidth={2.4} /></span>
          </div>
        </div>
        <div className="flex w-[150px] items-center gap-2"><Icon name="volume" size={15} strokeWidth={2} className="text-txt-mute" /><Slider value={volume} onChange={setVolume} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`} /></div>
        <div className="flex-1" />
        <div className="flex items-center gap-1 rounded-md bg-panel-2 p-0.5">
          <button onClick={toggleLoop} title={selectedSegment ? "Loop selected segment" : "Loop full audio"} className={cn("flex h-7 w-[30px] items-center justify-center rounded", loop ? "bg-blue-500 text-white" : "text-txt-dim")}><Icon name="repeat" size={14} strokeWidth={2.2} /></button>
          <button onClick={onDownload} title="Download full audio" className="flex h-7 w-7 items-center justify-center rounded text-txt-dim hover:bg-panel-3 hover:text-txt"><Icon name="download" size={14} strokeWidth={2.2} /></button>
        </div>
        <div className="flex items-center gap-1"><TransportButton icon="zoom-out" title="Zoom out" onClick={zoomOut} /><TransportButton icon="zoom-in" title="Zoom in" onClick={zoomIn} /></div>
      </div>
    </div>
  );
}
