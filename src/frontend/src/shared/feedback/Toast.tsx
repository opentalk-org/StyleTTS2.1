import { create } from "zustand";

import { Icon } from "../icons";
import type { Tone } from "../ui/Badge";

type ToastTone = "success" | "error";

type ToastState = {
  id: number;
  message: string;
  sub?: string;
  tone: ToastTone;
} | null;

type ToastStore = {
  toast: ToastState;
  show: (message: string, sub?: string, tone?: ToastTone) => void;
};

let timer: ReturnType<typeof setTimeout> | undefined;

export const useToast = create<ToastStore>((set) => ({
  toast: null,
  show: (message, sub, tone = "success") => {
    if (timer) clearTimeout(timer);
    set({ toast: { id: Date.now(), message, sub, tone } });
    timer = setTimeout(() => set({ toast: null }), 3200);
  },
}));

export function showToast(message: string, sub?: string, tone?: ToastTone) {
  useToast.getState().show(message, sub, tone);
}

const TONE: Record<ToastTone, { icon: "check-circle" | "x-circle"; color: Tone }> = {
  success: { icon: "check-circle", color: "emerald" },
  error: { icon: "x-circle", color: "red" },
};

export function ToastHost() {
  const toast = useToast((s) => s.toast);
  if (!toast) return null;
  const t = TONE[toast.tone];
  return (
    <div
      className="fixed bottom-6 left-1/2 z-[200] -translate-x-1/2"
      style={{ animation: "fadein 200ms ease" }}
    >
      <div className="flex items-center gap-2.5 rounded-[10px] border border-line bg-panel px-4 py-3 shadow-[0_8px_24px_rgba(17,24,39,0.14)]">
        <span className={t.color === "red" ? "text-red-500" : "text-emerald-600"}>
          <Icon name={t.icon} size={17} strokeWidth={2.2} />
        </span>
        <span className="text-[13px] font-semibold text-txt">{toast.message}</span>
        {toast.sub ? (
          <span className="font-mono text-xs tabular-nums text-txt-mute">{toast.sub}</span>
        ) : null}
      </div>
    </div>
  );
}
