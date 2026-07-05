import type { ReactNode } from "react";

import { Card } from "@/shared/ui/Card";

/** Titled form section with a small uppercase eyebrow tag above the title. */
export function FormSection({
  title,
  tag,
  children,
}: {
  title: string;
  tag: string;
  children: ReactNode;
}) {
  return (
    <Card className="px-5 py-[18px]">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <div className="text-[15px] font-bold tracking-tight text-txt">
          {title}
        </div>
        <div className="text-[11px] font-bold uppercase tracking-[0.08em] text-blue-500">
          {tag}
        </div>
      </div>
      {children}
    </Card>
  );
}
