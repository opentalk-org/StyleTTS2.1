import { useEffect, useRef, useState } from "react";

import { fmtClock } from "../format";
import { Icon } from "../icons";
import { showToast } from "../feedback/Toast";
import { WaveformBars } from "./WaveformBars";

/**
 * Reusable audio player: play/pause, a progress waveform, a time readout, and a
 * download action. Playback is mocked with a timer (no real audio in scaffold).
 */
export function WaveformPlayer({
  seed,
  duration,
  fileName = "sample.wav",
}: {
  seed: number;
  duration: number;
  fileName?: string;
}) {
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  useEffect(() => {
    if (!playing) return;
    timer.current = setInterval(() => {
      setPos((p) => {
        if (p + 0.05 >= duration) {
          setPlaying(false);
          return 0;
        }
        return p + 0.05;
      });
    }, 50);
    return () => clearInterval(timer.current);
  }, [playing, duration]);

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={() => setPlaying((v) => !v)}
        className="flex h-[38px] w-[38px] flex-none items-center justify-center rounded-full border-0 bg-blue-500 text-white cursor-pointer"
      >
        <Icon name={playing ? "pause" : "play"} size={16} strokeWidth={2.2} />
      </button>
      <WaveformBars seed={seed} bars={64} progress={playing ? pos / duration : 0} className="flex-1" />
      <span className="min-w-[74px] text-right font-mono text-[11.5px] tabular-nums text-txt-mute">
        {fmtClock(playing ? pos : 0)} / {fmtClock(duration)}
      </span>
      <button
        onClick={() => showToast(`Downloading ${fileName}`)}
        title="Download"
        className="flex h-[34px] w-[34px] flex-none items-center justify-center rounded-md border-0 bg-panel-2 text-txt-dim cursor-pointer"
      >
        <Icon name="download" size={16} strokeWidth={2.2} />
      </button>
    </div>
  );
}
