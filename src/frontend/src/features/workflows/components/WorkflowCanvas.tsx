import { useEffect, useRef, useState, type PointerEvent } from "react";

import { WorkflowBottomBar } from "./WorkflowBottomBar";
import { WorkflowEdges } from "./WorkflowEdges";
import { WorkflowNodeCard } from "./WorkflowNodeCard";
import { WorkflowRunPanel } from "./WorkflowRunPanel";
import { useWorkflowStore } from "../store";
import { graphPoint } from "../logic";

export function WorkflowCanvas() {
  const canvasRef = useRef<HTMLElement | null>(null);
  const [panning, setPanning] = useState<{ x: number; y: number } | null>(null);
  const [marquee, setMarquee] = useState<{ x: number; y: number; startX: number; startY: number } | null>(null);
  const { graph, viewport, wireDraft, selectNode, selectNodes, panViewport, zoomAt, setWireDraft, deleteSelection } = useWorkflowStore();

  useEffect(() => {
    const onKeydown = (event: KeyboardEvent) => {
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (event.key === "Escape") {
        selectNode(null);
        setWireDraft(null);
      }
      if (event.key === "Delete" || event.key === "Backspace") deleteSelection();
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [deleteSelection, selectNode, setWireDraft]);

  const onPointerMove = (event: PointerEvent) => {
    if (panning) {
      panViewport(event.clientX - panning.x, event.clientY - panning.y);
      setPanning({ x: event.clientX, y: event.clientY });
    }
    if (marquee && canvasRef.current) {
      const box = canvasRef.current.getBoundingClientRect();
      const point = graphPoint(viewport, event.clientX, event.clientY, box.left, box.top);
      setMarquee({ ...marquee, x: point.x, y: point.y });
    }
    if (wireDraft && canvasRef.current) {
      const box = canvasRef.current.getBoundingClientRect();
      const point = graphPoint(viewport, event.clientX, event.clientY, box.left, box.top);
      setWireDraft({ ...wireDraft, x: point.x, y: point.y });
    }
  };

  return (
    <section
      ref={canvasRef}
      className="relative min-w-0 flex-1 overflow-hidden bg-app"
      onClick={() => selectNode(null)}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        if (event.shiftKey) {
          const box = event.currentTarget.getBoundingClientRect();
          const point = graphPoint(viewport, event.clientX, event.clientY, box.left, box.top);
          setMarquee({ x: point.x, y: point.y, startX: point.x, startY: point.y });
        } else {
          setPanning({ x: event.clientX, y: event.clientY });
        }
      }}
      onPointerMove={onPointerMove}
      onPointerUp={() => {
        if (marquee) {
          const left = Math.min(marquee.startX, marquee.x);
          const right = Math.max(marquee.startX, marquee.x);
          const top = Math.min(marquee.startY, marquee.y);
          const bottom = Math.max(marquee.startY, marquee.y);
          selectNodes(graph.nodes.filter((node) => node.x >= left && node.x <= right && node.y >= top && node.y <= bottom).map((node) => node.id));
        }
        setPanning(null);
        setMarquee(null);
      }}
      onWheel={(event) => {
        event.preventDefault();
        const box = event.currentTarget.getBoundingClientRect();
        const next = viewport.zoom * (event.deltaY < 0 ? 1.12 : 0.88);
        zoomAt(next, event.clientX - box.left, event.clientY - box.top);
      }}
    >
      <div className="absolute inset-0 bg-[linear-gradient(#e5e7eb_1px,transparent_1px),linear-gradient(90deg,#e5e7eb_1px,transparent_1px)] bg-[size:32px_32px]" />
      <WorkflowEdges />
      <div
        onClick={(event) => event.stopPropagation()}
        style={{ transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`, transformOrigin: "0 0" }}
      >
        {graph.nodes.map((node) => <WorkflowNodeCard key={node.id} node={node} />)}
      </div>
      {marquee ? (
        <div
          className="pointer-events-none absolute border border-blue-500 bg-blue-500/10"
          style={{
            left: Math.min(marquee.startX, marquee.x) * viewport.zoom + viewport.x,
            top: Math.min(marquee.startY, marquee.y) * viewport.zoom + viewport.y,
            width: Math.abs(marquee.x - marquee.startX) * viewport.zoom,
            height: Math.abs(marquee.y - marquee.startY) * viewport.zoom,
          }}
        />
      ) : null}
      <WorkflowRunPanel />
      <WorkflowBottomBar />
    </section>
  );
}
