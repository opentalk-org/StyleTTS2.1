import type { ComponentType } from "react";

import { AudioScreen } from "@/features/audio/AudioScreen";
import { SegmentEditor } from "@/features/audio/SegmentEditor";
import { CheckpointsScreen } from "@/features/checkpoints/CheckpointsScreen";
import { ClusterScreen } from "@/features/cluster/ClusterScreen";
import { DatasetsScreen } from "@/features/datasets/DatasetsScreen";
import { JobsScreen } from "@/features/jobs/JobsScreen";
import { RunsScreen } from "@/features/runs/RunsScreen";
import { SettingsScreen } from "@/features/settings/SettingsScreen";
import { StatisticsScreen } from "@/features/statistics/StatisticsScreen";
import { TestingScreen } from "@/features/testing/TestingScreen";
import { TrainingScreen } from "@/features/training/TrainingScreen";
import { VoicesScreen } from "@/features/voices/VoicesScreen";
import { WorkflowsScreen } from "@/features/workflows/WorkflowsScreen";
import type { Screen } from "./navStore";
import { useNav } from "./navStore";

const SCREENS: Record<Screen, ComponentType> = {
  datasets: DatasetsScreen,
  voices: VoicesScreen,
  audio: AudioScreen,
  editor: SegmentEditor,
  statistics: StatisticsScreen,
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
  return <Active />;
}
