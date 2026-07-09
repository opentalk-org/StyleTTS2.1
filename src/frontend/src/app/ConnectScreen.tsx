import { Icon } from "@/shared/icons";
import { useNav } from "./navStore";

export function ConnectScreen() {
  const { backendUrl, setBackendUrl, connect } = useNav();

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden bg-app p-6">
      <div className="absolute -top-40 -right-28 h-[560px] w-[560px] rounded-full bg-blue-500 opacity-10" />
      <div className="absolute -bottom-28 -left-20 h-[380px] w-[380px] rotate-[18deg] rounded-[18px] bg-emerald-500 opacity-[0.08]" />
      <div className="absolute left-[13%] top-16 h-40 w-40 rounded-full bg-amber-500 opacity-[0.12]" />

      <div className="relative flex w-full max-w-[440px] flex-col">
        <div className="mb-7 flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-[10px] bg-blue-500 text-white">
            <Icon name="audio-lines" size={26} strokeWidth={2.4} />
          </div>
          <div className="text-[22px] font-extrabold tracking-tight text-txt">
            StyleTTS <span className="text-blue-500">Studio</span>
          </div>
        </div>
        <div className="mb-2.5 text-4xl font-extrabold leading-tight tracking-tight text-txt">
          Connect your lab.
        </div>
        <div className="mb-7 text-[15px] leading-relaxed text-txt-dim">
          Point the workbench at your training backend. Everything — datasets, jobs,
          checkpoints — streams from there.
        </div>
        <label className="mb-[18px] flex flex-col gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-txt-dim">
            Backend URL
          </span>
          <input
            value={backendUrl}
            onChange={(e) => setBackendUrl(e.target.value)}
            placeholder="http://127.0.0.1:8000"
            className="h-13 rounded-lg border-2 border-line-2 bg-panel px-4 text-base text-txt outline-none focus:border-blue-500"
          />
        </label>
        <button
          onClick={connect}
          className="flex h-13 items-center justify-center gap-2 rounded-lg border-0 bg-blue-500 text-base font-semibold text-white cursor-pointer transition-transform hover:scale-[1.02] hover:bg-blue-600"
        >
          Enter workbench
          <Icon name="arrow-left" size={20} strokeWidth={2.4} className="rotate-180" />
        </button>
        <div className="mt-[22px] flex items-center gap-2 text-xs text-txt-mute">
          <Icon name="server" size={14} strokeWidth={2} />
          Single sign-on coming soon — for now this stays on your machine.
        </div>
      </div>
    </div>
  );
}
