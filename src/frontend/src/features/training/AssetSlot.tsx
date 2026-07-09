import { Icon } from "@/shared/icons";
import { Select, type Option } from "@/shared/ui/Select";

export function AssetSlot({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2.5 rounded-lg bg-panel-2 p-3">
      <div className="text-xs font-semibold text-txt-dim">{label}</div>
      <div className="flex items-center gap-2">
        <Icon name="box" size={16} className="text-emerald-600" />
        <Select value={value} onChange={onChange} options={options} />
      </div>
    </div>
  );
}
