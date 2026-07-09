import { create } from "zustand";

import { Icon, type IconName } from "../icons";
import { Button } from "../ui/Button";
import { cn } from "../ui/cn";

export type ConfirmConfig = {
  title: string;
  desc: string;
  danger?: boolean;
  label?: string;
  icon?: IconName;
  onConfirm: () => void;
};

type ConfirmStore = {
  config: ConfirmConfig | null;
  ask: (config: ConfirmConfig) => void;
  close: () => void;
};

export const useConfirm = create<ConfirmStore>((set) => ({
  config: null,
  ask: (config) => set({ config }),
  close: () => set({ config: null }),
}));

export function askConfirm(config: ConfirmConfig) {
  useConfirm.getState().ask(config);
}

export function ConfirmHost() {
  const { config, close } = useConfirm();
  if (!config) return null;
  const icon = config.icon ?? (config.danger ? "alert" : "check-circle");
  return (
    <div
      onClick={close}
      className="fixed inset-0 z-[320] flex items-center justify-center bg-gray-900/45 p-6"
      style={{ animation: "fadein 140ms ease" }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-[420px] overflow-hidden rounded-xl bg-panel"
      >
        <div className="flex gap-3.5 px-[22px] pt-[22px] pb-[18px]">
          <div
            className={cn(
              "flex h-10 w-10 flex-none items-center justify-center rounded-[10px]",
              config.danger ? "bg-red-50 text-red-500" : "bg-blue-50 text-blue-600",
            )}
          >
            <Icon name={icon} size={20} strokeWidth={2.2} />
          </div>
          <div className="flex-1">
            <div className="mb-1 text-base font-bold text-txt">{config.title}</div>
            <div className="text-[13px] leading-normal text-txt-dim">{config.desc}</div>
          </div>
        </div>
        <div className="flex justify-end gap-2.5 border-t border-line bg-panel-2 px-[22px] py-4">
          <Button variant="secondary" onClick={close}>
            Cancel
          </Button>
          <Button
            variant={config.danger ? "danger" : "primary"}
            icon={config.danger ? "trash" : "check"}
            onClick={() => {
              config.onConfirm();
              close();
            }}
          >
            {config.label ?? "Confirm"}
          </Button>
        </div>
      </div>
    </div>
  );
}
