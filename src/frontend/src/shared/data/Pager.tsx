import { Icon } from "../icons";
import { cn } from "../ui/cn";

export function Pager({
  page,
  pages,
  onChange,
}: {
  page: number;
  pages: number;
  onChange: (page: number) => void;
}) {
  const go = (p: number) => onChange(Math.max(0, Math.min(pages - 1, p)));
  const nums: (number | "…")[] = [];
  for (let i = 0; i < pages; i++) {
    if (i === 0 || i === pages - 1 || (i >= page - 1 && i <= page + 1)) nums.push(i);
    else if (nums[nums.length - 1] !== "…") nums.push("…");
  }

  const arrow = (target: number, disabled: boolean, flip: boolean) => (
    <button
      disabled={disabled}
      onClick={() => go(target)}
      className="flex h-8 w-8 items-center justify-center rounded-md border border-line bg-panel text-txt-dim cursor-pointer transition-colors enabled:hover:border-line-2 enabled:hover:text-txt disabled:cursor-default disabled:opacity-45"
    >
      <span className={cn("flex", flip && "rotate-180")}>
        <Icon name="arrow-left" size={15} strokeWidth={2.4} />
      </span>
    </button>
  );

  return (
    <div className="flex items-center gap-1.5">
      {arrow(page - 1, page <= 0, false)}
      <div className="flex items-center gap-0.5 rounded-[9px] border border-line bg-panel p-[3px]">
        {nums.map((n, i) =>
          n === "…" ? (
            <span key={`e${i}`} className="w-6 select-none text-center text-[13px] text-txt-mute">
              …
            </span>
          ) : (
            <button
              key={n}
              onClick={() => go(n)}
              className={cn(
                "flex h-8 min-w-8 items-center justify-center rounded-md px-1.5 text-[13px] font-bold tabular-nums transition-colors",
                n === page
                  ? "bg-blue-500 text-white cursor-default"
                  : "bg-transparent text-txt-dim cursor-pointer hover:bg-panel-2",
              )}
            >
              {n + 1}
            </button>
          ),
        )}
      </div>
      {arrow(page + 1, page >= pages - 1, true)}
    </div>
  );
}
