import type { ReactNode } from "react";

import { Card } from "@/shared/ui/Card";

/**
 * A titled panel wrapping a single chart. `unit` renders as a small uppercase
 * caption; `span` makes the card span a full grid row.
 */
export function ChartCard({
  title,
  unit,
  span,
  children,
}: {
  title: string;
  unit?: string;
  span?: boolean;
  children: ReactNode;
}) {
  return (
    <Card className={`px-4 pt-[14px] pb-3 ${span ? "col-span-full" : ""}`}>
      <div className="mb-3 flex items-baseline justify-between">
        <div className="text-[12.5px] font-bold text-txt">{title}</div>
        {unit ? (
          <div className="text-[10.5px] font-semibold uppercase tracking-wide text-txt-mute">
            {unit}
          </div>
        ) : null}
      </div>
      {children}
    </Card>
  );
}
