import { useEffect, useRef, useState } from "react";

import { fmtClock } from "../format";
import { Icon } from "../icons";
import { showToast } from "../feedback/Toast";
import { WaveformBars } from "./WaveformBars";

export function WaveformPlayer({
  seed,
  duration,
  fileName = "sample.wav",
  src,
  clipStart = 0,
  clipEnd,
}: {
  seed: number;
  duration: number;
  fileName?: string;
  src?: string;
  clipStart?: number;
  clipEnd?: number;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  useEffect(() => {
    if (src || !playing) return;
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
  }, [playing, duration, src]);

  useEffect(() => {
    const element = audioRef.current;
    if (!src || !element) return;
    const end = clipEnd ?? clipStart + duration;
    const onTime = () => {
      if (element.currentTime >= end) {
        element.pause();
        element.currentTime = clipStart;
        setPlaying(false);
        setPos(0);
        return;
      }
      setPos(Math.max(0, element.currentTime - clipStart));
    };
    const onEnd = () => {
      setPlaying(false);
      setPos(0);
      element.currentTime = clipStart;
    };
    element.addEventListener("timeupdate", onTime);
    element.addEventListener("ended", onEnd);
    return () => {
      element.removeEventListener("timeupdate", onTime);
      element.removeEventListener("ended", onEnd);
    };
  }, [clipEnd, clipStart, duration, src]);

  const toggle = () => {
    if (!src) {
      setPlaying((v) => !v);
      return;
    }
    const element = audioRef.current;
    if (!element) return;
    if (playing) {
      element.pause();
      setPlaying(false);
    } else {
      const end = clipEnd ?? clipStart + duration;
      if (element.currentTime < clipStart || element.currentTime >= end) element.currentTime = clipStart;
      void element
        .play()
        .then(() => setPlaying(true))
        .catch(() => showToast("Could not play audio", undefined, "error"));
    }
  };

  const download = () => {
    if (!src) {
      showToast(`Downloading ${fileName}`);
      return;
    }
    const anchor = document.createElement("a");
    anchor.href = src;
    anchor.download = fileName;
    anchor.click();
  };

  const total = duration || 0;

  return (
    <div className="flex items-center gap-3">
      {src ? <audio ref={audioRef} src={src} preload="metadata" /> : null}
      <button
        onClick={toggle}
        className="flex h-[38px] w-[38px] flex-none items-center justify-center rounded-full border-0 bg-blue-500 text-white cursor-pointer"
      >
        <Icon name={playing ? "pause" : "play"} size={16} strokeWidth={2.2} />
      </button>
      <WaveformBars seed={seed} bars={64} progress={total ? pos / total : 0} className="flex-1" />
      <span className="min-w-[74px] text-right font-mono text-[11.5px] tabular-nums text-txt-mute">
        {fmtClock(pos)} / {fmtClock(total)}
      </span>
      <button
        onClick={download}
        title="Download"
        className="flex h-[34px] w-[34px] flex-none items-center justify-center rounded-md border-0 bg-panel-2 text-txt-dim cursor-pointer"
      >
        <Icon name="download" size={16} strokeWidth={2.2} />
      </button>
    </div>
  );
}
