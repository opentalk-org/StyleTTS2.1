import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/app/AppShell";
import { ArtifactsScreen } from "@/features/artifacts/ArtifactsScreen";
import { AudioScreen } from "@/features/audio/AudioScreen";
import { SegmentEditor } from "@/features/audio/SegmentEditor";
import { CheckpointsScreen } from "@/features/checkpoints/CheckpointsScreen";
import { ClusterScreen } from "@/features/cluster/ClusterScreen";
import { DatasetsScreen } from "@/features/datasets/DatasetsScreen";
import { JobsScreen } from "@/features/jobs/JobsScreen";
import { MosScreen } from "@/features/mos/MosScreen";
import { RunsScreen } from "@/features/runs/RunsScreen";
import { SettingsScreen } from "@/features/settings/SettingsScreen";
import { SpeakersScreen } from "@/features/speakers/SpeakersScreen";
import { TestingScreen } from "@/features/testing/TestingScreen";
import { TrainingScreen } from "@/features/training/TrainingScreen";
import { WorkflowsScreen } from "@/features/workflows/WorkflowsScreen";

const StatisticsScreen = lazy(() =>
  import("@/features/statistics/StatisticsScreen").then((module) => ({ default: module.StatisticsScreen })),
);

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/training" replace />} />
        <Route path="datasets" element={<DatasetsScreen />} />
        <Route path="speakers" element={<SpeakersScreen />} />
        <Route path="audio" element={<AudioScreen />} />
        <Route path="audio/:audioFileId" element={<SegmentEditor />} />
        <Route path="mos" element={<MosScreen />} />
        <Route path="statistics" element={<Suspense fallback={<div className="px-7 pt-10 text-[13px] text-txt-mute">Loading…</div>}><StatisticsScreen /></Suspense>} />
        <Route path="artifacts" element={<ArtifactsScreen />} />
        <Route path="workflows" element={<WorkflowsScreen />} />
        <Route path="checkpoints" element={<CheckpointsScreen />} />
        <Route path="training" element={<TrainingScreen />} />
        <Route path="runs" element={<RunsScreen />} />
        <Route path="testing" element={<TestingScreen />} />
        <Route path="cluster" element={<ClusterScreen />} />
        <Route path="jobs" element={<JobsScreen />} />
        <Route path="settings" element={<SettingsScreen />} />
        <Route path="*" element={<Navigate to="/training" replace />} />
      </Route>
    </Routes>
  );
}
