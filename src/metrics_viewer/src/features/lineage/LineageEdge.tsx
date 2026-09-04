import { BaseEdge, type EdgeProps } from "@xyflow/react";

import { forkPath } from "./logic";

export interface LineageEdgeData extends Record<string, unknown> {
  /** x of the vertical segment; each fork that shares rows with another gets its own. */
  channelX: number;
  fork: boolean;
  color: string;
  dimmed: boolean;
  highlighted: boolean;
}

/**
 * A resume link. Same-run links are a plain horizontal line; a fork leaves its source,
 * travels down its own channel between the two columns, and turns into the target — so
 * two branches out of the same run never share a segment.
 */
export function LineageEdge({ id, sourceX, sourceY, targetX, targetY, data }: EdgeProps) {
  const { channelX, fork, color, dimmed, highlighted } = data as unknown as LineageEdgeData;
  const path = fork
    ? forkPath(sourceX, sourceY, channelX, targetX, targetY)
    : `M ${sourceX},${sourceY} L ${targetX},${targetY}`;
  return (
    <BaseEdge
      id={id}
      path={path}
      style={{
        stroke: color,
        strokeWidth: highlighted ? 2.5 : fork ? 1.75 : 2,
        strokeDasharray: fork ? "5 4" : undefined,
        strokeLinecap: "round",
        opacity: dimmed ? 0.25 : 1,
      }}
    />
  );
}
