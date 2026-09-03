import { useEffect, useRef } from "react";

import { useQueryClient } from "@tanstack/react-query";

import { backendWebSocketUrl } from "@/app/backend";
import { useAppStore } from "@/app/store";
import { AUDIO_FILES_KEY } from "@/features/audio/query";
import { DATASETS_KEY } from "@/features/datasets/query";

import { useWorkflowStore } from "./store";
import type { RunStatus, WorkflowSocketMessage } from "./types";

const TERMINAL_STATES = new Set(["succeeded", "failed", "stopped"]);

export function useWorkflowSocket() {
  const backendUrl = useAppStore((state) => state.backendUrl);
  const activeRunId = useWorkflowStore((state) => state.activeRunId);
  const applyRunnerStatus = useWorkflowStore((state) => state.applyRunnerStatus);
  const applyRunStatus = useWorkflowStore((state) => state.applyRunStatus);
  const applyRunSnapshot = useWorkflowStore((state) => state.applyRunSnapshot);
  const queryClient = useQueryClient();
  const socketRef = useRef<WebSocket | null>(null);
  const finishedRuns = useRef<Set<string>>(new Set());

  useEffect(() => {
    // A finished run may have written audio segments/records, so refresh the
    // audio and dataset caches once per run when it reaches a terminal state.
    const onRunStatus = (status: RunStatus) => {
      applyRunStatus(status);
      if (!TERMINAL_STATES.has(status.state) || finishedRuns.current.has(status.run_id)) return;
      finishedRuns.current.add(status.run_id);
      queryClient.invalidateQueries({ queryKey: [AUDIO_FILES_KEY] });
      queryClient.invalidateQueries({ queryKey: [DATASETS_KEY] });
    };
    const socket = new WebSocket(backendWebSocketUrl("/ws", backendUrl));
    socketRef.current = socket;
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data) as WorkflowSocketMessage;
      if (message.type === "runner_status") applyRunnerStatus(message.status.runs);
      if (message.type === "run_status") onRunStatus(message.status);
      if (message.type === "run_snapshot") {
        onRunStatus(message.status);
        applyRunSnapshot(message.run_id, message.snapshot);
      }
    });
    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [applyRunnerStatus, applyRunSnapshot, applyRunStatus, backendUrl, queryClient]);

  useEffect(() => {
    const socket = socketRef.current;
    if (!socket || !activeRunId) return;
    const message = JSON.stringify({ type: "watch_run", run_id: activeRunId });
    if (socket.readyState === WebSocket.OPEN) socket.send(message);
    else socket.addEventListener("open", () => socket.send(message), { once: true });
  }, [activeRunId]);
}
