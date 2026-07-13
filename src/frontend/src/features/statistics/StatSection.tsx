import type { ReactNode } from "react";

// One labeled band of the statistics page. Every group of charts gets the same header
// treatment (eyebrow title + optional caption) and the same spacing, so the page reads as
// a sequence of deliberate sections instead of a wall of cards.
export function StatSection({
  title,
  caption,
  actions,
  children,
}: {
  title: string;
  caption?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section>
      <div className="mb-3.5 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h3 className="text-[12px] font-bold uppercase tracking-[0.08em] text-txt-dim">{title}</h3>
          {caption ? <p className="mt-1 text-[11.5px] leading-snug text-txt-mute">{caption}</p> : null}
        </div>
        {actions ? <div className="flex flex-none items-center gap-2.5">{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

// Column counts are looked up (not interpolated) so Tailwind keeps them in the build.
export const GRID_COLS: Record<number, string> = {
  1: "grid-cols-1",
  2: "grid-cols-2",
  3: "grid-cols-3",
  4: "grid-cols-4",
};
