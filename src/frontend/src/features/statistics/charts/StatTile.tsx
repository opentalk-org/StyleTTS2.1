import { TILE_TEXT_CLASS, type Tone } from "../logic";
import { Card } from "@/shared/ui/Card";

export function StatTile({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone: Tone;
}) {
  return (
    <Card className="px-4 py-[14px]">
      <div className={`text-[24px] font-extrabold tracking-tight tabular-nums ${TILE_TEXT_CLASS[tone]}`}>
        {value}
      </div>
      <div className="mt-[2px] text-[12px] font-semibold text-txt">{label}</div>
      {sub ? <div className="mt-[1px] text-[11px] text-txt-mute">{sub}</div> : null}
    </Card>
  );
}
