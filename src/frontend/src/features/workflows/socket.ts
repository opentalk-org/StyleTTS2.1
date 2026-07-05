import { useEffect, useRef } from "react";

import { useWorkflowStore } from "./store";
import type { WorkflowSocketMessage } from "./types";

function socketUrl(): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws`;
}

export function useWorkflowSocket() {
  const activeRunId = useWorkflowStore((state) => state.activeRunId);
  const applyRunnerStatus = useWorkflowStore((state) => state.applyRunnerStatus);
  const applyRunStatus = useWorkflowStore((state) => state.applyRunStatus);
  const applyRunSnapshot = useWorkflowStore((state) => state.applyRunSnapshot);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const socket = new WebSocket(socketUrl());
    socketRef.current = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data) as WorkflowSocketMessage;
      if (message.type === "runner_status") applyRunnerStatus(message.status.runs);
      if (message.type === "run_status") applyRunStatus(message.status);
      if (message.type === "run_snapshot") {
        applyRunStatus(message.status);
        applyRunSnapshot(message.run_id, message.snapshot);
      }
    });
    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [applyRunnerStatus, applyRunSnapshot, applyRunStatus]);

  useEffect(() => {
    const socket = socketRef.current;
    if (!socket || !activeRunId) return;
    const message = JSON.stringify({ type: "watch_run", run_id: activeRunId });
    if (socket.readyState === WebSocket.OPEN) socket.send(message);
    else socket.addEventListener("open", () => socket.send(message), { once: true });
  }, [activeRunId]);
}
