import { type ComponentType, lazy, Suspense } from "react";

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
import { TestingScreen } from "@/features/testing/TestingScreen";
import { TrainingScreen } from "@/features/training/TrainingScreen";
import { SpeakersScreen } from "@/features/speakers/SpeakersScreen";
import { WorkflowsScreen } from "@/features/workflows/WorkflowsScreen";
import type { Screen } from "./navStore";
import { useNav } from "./navStore";

// The statistics screen pulls in Plotly (~3.5 MB), so it is code-split and only fetched when
// the tab is first opened, keeping it out of the initial bundle.
const StatisticsScreen = lazy(() =>
  import("@/features/statistics/StatisticsScreen").then((m) => ({ default: m.StatisticsScreen })),
);

const SCREENS: Record<Screen, ComponentType> = {
  datasets: DatasetsScreen,
  speakers: SpeakersScreen,
  audio: AudioScreen,
  editor: SegmentEditor,
  mos: MosScreen,
  statistics: StatisticsScreen,
  artifacts: ArtifactsScreen,
  workflows: WorkflowsScreen,
  checkpoints: CheckpointsScreen,
  training: TrainingScreen,
  runs: RunsScreen,
  testing: TestingScreen,
  cluster: ClusterScreen,
  jobs: JobsScreen,
  settings: SettingsScreen,
};

export function ScreenRouter() {
  const screen = useNav((s) => s.screen);
  const Active = SCREENS[screen];
  return (
    <Suspense fallback={<div className="px-7 pt-10 text-[13px] text-txt-mute">Loading…</div>}>
      <Active />
    </Suspense>
  );
}
