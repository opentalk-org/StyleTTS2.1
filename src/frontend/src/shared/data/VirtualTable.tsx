import { useVirtualizer } from "@tanstack/react-virtual";
import { type ReactNode, useState } from "react";

import { cn } from "../ui/cn";

/**
 * Windowed list: only the visible rows are in the DOM, so `count` can be in the
 * millions. Rows are absolutely positioned and measured, so they may vary in
 * height (e.g. an expanded row). `renderRow(index)` must render exactly one row.
 */
export function VirtualTable({
  count,
  estimateRowHeight,
  renderRow,
  header,
  overscan = 12,
  className,
  scrollClassName,
  pageScroll = false,
}: {
  count: number;
  estimateRowHeight: number;
  renderRow: (index: number) => ReactNode;
  header?: ReactNode;
  overscan?: number;
  className?: string;
  scrollClassName?: string;
  pageScroll?: boolean;
}) {
  const [parentElement, setParentElement] = useState<HTMLDivElement | null>(null);
  const pageScrollElement = pageScroll ? parentElement?.closest("main") : null;
  const scrollMargin = pageScroll && parentElement && pageScrollElement
    ? parentElement.getBoundingClientRect().top - pageScrollElement.getBoundingClientRect().top + pageScrollElement.scrollTop
    : 0;
  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => pageScrollElement ?? parentElement,
    estimateSize: () => estimateRowHeight,
    overscan,
    scrollMargin,
  });

  return (
    <div className={cn(pageScroll ? "" : "flex min-h-0 flex-col", className)}>
      {header}
      <div ref={setParentElement} className={cn(pageScroll ? "" : "min-h-0 flex-1 overflow-y-auto", scrollClassName)}>
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((item) => (
            <div
              key={item.key}
              data-index={item.index}
              ref={virtualizer.measureElement}
              className="absolute left-0 top-0 w-full"
              style={{ transform: `translateY(${item.start - scrollMargin}px)` }}
            >
              {renderRow(item.index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
