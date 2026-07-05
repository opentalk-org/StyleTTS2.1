import { Icon } from "@/shared/icons";

/** Muted card showing an optional pretrained asset file for a training run. */
export function AssetSlot({ label, file }: { label: string; file: string }) {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg bg-panel-2 p-3">
      <div className="text-xs font-semibold text-txt-dim">{label}</div>
      <div className="flex items-center gap-2">
        <Icon name="box" size={16} className="text-emerald-600" />
        <span className="truncate font-mono text-[13px] font-semibold text-txt">
          {file}
        </span>
      </div>
    </div>
  );
}
