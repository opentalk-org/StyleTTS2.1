import type { IconName } from "@/shared/icons";
import type { Screen } from "./navStore";

export type NavItem = { id: Screen; label: string; icon: IconName };

export const NAV_ITEMS: NavItem[] = [
  { id: "datasets", label: "Datasets", icon: "database" },
  { id: "voices", label: "Voices", icon: "mic" },
  { id: "audio", label: "Audio Files", icon: "audio-lines" },
  { id: "statistics", label: "Statistics", icon: "bar-chart" },
  { id: "artifacts", label: "Artifacts", icon: "sparkles" },
  { id: "workflows", label: "Workflows", icon: "workflow" },
  { id: "checkpoints", label: "Checkpoints", icon: "box" },
  { id: "training", label: "Training", icon: "sliders" },
  { id: "runs", label: "Runs", icon: "activity" },
  { id: "testing", label: "Testing", icon: "flask" },
  { id: "cluster", label: "Cluster", icon: "server" },
  { id: "jobs", label: "Jobs", icon: "list-checks" },
  { id: "settings", label: "Settings", icon: "settings" },
];

export const SCREEN_META: Record<Screen, { title: string; icon: IconName }> = {
  datasets: { title: "Datasets", icon: "database" },
  voices: { title: "Voices", icon: "mic" },
  audio: { title: "Audio Files", icon: "audio-lines" },
  editor: { title: "Segment Editor", icon: "audio-lines" },
  statistics: { title: "Statistics", icon: "bar-chart" },
  artifacts: { title: "Artifacts", icon: "sparkles" },
  workflows: { title: "Workflows", icon: "workflow" },
  checkpoints: { title: "Checkpoints", icon: "box" },
  training: { title: "Training", icon: "sliders" },
  runs: { title: "Runs", icon: "activity" },
  testing: { title: "Testing", icon: "flask" },
  cluster: { title: "Cluster", icon: "server" },
  jobs: { title: "Jobs", icon: "list-checks" },
  settings: { title: "Settings", icon: "settings" },
};
