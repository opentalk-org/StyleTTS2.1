/**
 * Domain types for the mock layer. Loosely mirror the IDEA.md DB sketch
 * (audio_file, dataset, voice, checkpoint, …). These are the shapes a real
 * backend would return; features consume them through each feature's api.ts.
 */

export type AudioStatus = "raw" | "transcribed" | "normalized" | "denoised" | "flagged";

export type Segment = {
  id: string;
  start: number;
  end: number;
  text: string;
  phon: string;
  speaker: string;
};

export type AudioFile = {
  id: string;
  name: string;
  speaker: string;
  dur: number;
  sr: number;
  status: AudioStatus[];
  segments: number;
  updated: number;
  sizeMb: string;
  dataset: string | null;
};

export type Voice = {
  id: string;
  name: string;
  segments: number;
  datasets: string[];
};

export type Dataset = {
  id: string;
  name: string;
  files: number;
};

export type CheckpointType = "styletts2" | "asr" | "f0" | "plbert";

export type Checkpoint = {
  id: string;
  name: string;
  type: CheckpointType;
  job: string;
  spkMode: string;
  decoder: string;
  symbols: number;
  created: number;
};

export type StatEntry = {
  id: string;
  files: number;
  created: number;
};

export type StyleRef = {
  id: string;
  name: string;
  voice: string;
};
