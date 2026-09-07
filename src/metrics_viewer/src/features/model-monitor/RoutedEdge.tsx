import { BaseEdge, EdgeLabelRenderer, type EdgeProps } from "@xyflow/react";

import type { GraphEdgeData } from "./graph";
import type { Placement } from "./layout";

const CORNER = 8;

/**
 * Draws a wire along the lanes the layout reserved for it instead of straight at its target, which
 * is what keeps a dense stack from having its boxes crossed out by their own connections.
 */
export function RoutedEdge({ id, sourceX, sourceY, targetX, targetY, markerEnd, data }: EdgeProps) {
  const edge = data as GraphEdgeData;
  const points = [{ x: sourceX, y: sourceY }, ...edge.points, { x: targetX, y: targetY }];
  const middle = points[Math.floor(points.length / 2)];
  return (
    <>
      <BaseEdge
        id={id}
        path={orthogonalPath(points)}
        markerEnd={markerEnd}
        style={{
          stroke: edge.linked ? "var(--color-accent)" : edge.colour,
          strokeWidth: edge.linked ? 2 : 1.2,
          opacity: edge.dimmed ? 0.18 : edge.linked ? 1 : 0.6,
        }}
      />
      {edge.label.length === 0 || edge.dimmed ? null : (
        <EdgeLabelRenderer>
          <div
            className="pointer-events-none absolute rounded-sm border border-line bg-surface px-1 font-mono text-[10px] text-fg-muted"
            style={{ transform: `translate(-50%, -50%) translate(${middle.x}px, ${middle.y}px)` }}
          >
            {edge.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

/**
 * Drops straight down each column, then steps sideways at the gap the layout picked. Waypoints
 * already sit at those gaps, so every corner is a right angle rounded off.
 */
function orthogonalPath(points: Placement[]): string {
  const parts = [`M ${points[0].x},${points[0].y}`];
  for (let index = 0; index + 1 < points.length; index += 1) {
    const from = { x: points[index].x, y: points[index].y };
    const to = points[index + 1];
    if (Math.abs(to.x - from.x) < 1) {
      parts.push(`L ${to.x},${to.y}`);
      continue;
    }
    const radius = Math.min(CORNER, Math.abs(to.x - from.x) / 2, Math.abs(to.y - from.y));
    const direction = to.x > from.x ? 1 : -1;
    const vertical = to.y > from.y ? 1 : -1;
    parts.push(`L ${from.x},${to.y - vertical * radius}`);
    parts.push(`Q ${from.x},${to.y} ${from.x + direction * radius},${to.y}`);
    parts.push(`L ${to.x},${to.y}`);
  }
  return parts.join(" ");
}
