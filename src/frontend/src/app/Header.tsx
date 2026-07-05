import { Icon } from "@/shared/icons";
import { SCREEN_META } from "./nav";
import { useNav } from "./navStore";

export function Header() {
  const { screen, backendUrl } = useNav();
  const meta = SCREEN_META[screen];
  const backendShort = backendUrl.replace(/^https?:\/\//, "");

  return (
    <header className="flex h-14 flex-none items-center gap-4 border-b border-line bg-panel px-5">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="flex text-txt-mute">
          <Icon name={meta.icon} size={18} />
        </span>
        <div className="whitespace-nowrap text-[15px] font-bold tracking-tight text-txt">
          {meta.title}
        </div>
      </div>
      <div className="flex-1" />

      <div className="flex h-8 items-center gap-2 rounded-full bg-panel-2 px-3 text-xs font-medium text-txt-dim">
        <span className="h-[7px] w-[7px] rounded-full bg-emerald-500" />
        <span className="tabular-nums">{backendShort}</span>
      </div>
    </header>
  );
}
