import type { Artifact, MetricPoint, Project, Run } from "@/shared/types";

const NOW = Date.now();
const PROJECTS: Project[] = [
  { id: "voice-foundation", name: "Voice Foundation", description: "Multilingual acoustic model and latent flow experiments", createdAt: new Date(NOW - 8.6e9).toISOString(), lastRunAt: new Date(NOW - 7.2e5).toISOString(), runCount: 86, runningCount: 2 },
  { id: "decoder-lab", name: "Decoder Lab", description: "Vocoder quality, throughput, and discriminator studies", createdAt: new Date(NOW - 5.2e9).toISOString(), lastRunAt: new Date(NOW - 8.64e7).toISOString(), runCount: 41, runningCount: 0 },
  { id: "asr-alignment", name: "ASR Alignment", description: "CTC alignment and tokenizer sweeps", createdAt: new Date(NOW - 3.4e9).toISOString(), lastRunAt: new Date(NOW - 2.3e8).toISOString(), runCount: 29, runningCount: 1 },
  { id: "prosody-control", name: "Prosody Control", description: "Duration, pitch, and expressive style conditioning", createdAt: new Date(NOW - 7.1e9).toISOString(), lastRunAt: new Date(NOW - 1.9e7).toISOString(), runCount: 105, runningCount: 4 },
  { id: "speaker-encoder", name: "Speaker Encoder", description: "Cross-lingual speaker identity and embedding studies", createdAt: new Date(NOW - 6.2e9).toISOString(), lastRunAt: new Date(NOW - 4.4e7).toISOString(), runCount: 64, runningCount: 0 },
  { id: "data-ablation", name: "Data Ablations", description: "Dataset composition, filtering, and sampling experiments", createdAt: new Date(NOW - 2.8e9).toISOString(), lastRunAt: new Date(NOW - 3.2e8).toISOString(), runCount: 38, runningCount: 0 },
  { id: "latency-bench", name: "Inference Latency", description: "Real-time factor, memory, and batch throughput benchmarks", createdAt: new Date(NOW - 1.5e9).toISOString(), lastRunAt: new Date(NOW - 8.8e7).toISOString(), runCount: 22, runningCount: 1 },
  { id: "production-candidates", name: "Production Candidates", description: "Final quality gates and release candidate evaluations", createdAt: new Date(NOW - 9.2e8).toISOString(), lastRunAt: new Date(NOW - 9.4e6).toISOString(), runCount: 17, runningCount: 2 },
];

const STATUSES: Run["status"][] = ["succeeded", "succeeded", "succeeded", "running", "failed", "cancelled"];
const RUNS: Run[] = PROJECTS.flatMap((project, projectIndex) =>
  Array.from({ length: project.runCount }, (_, index) => {
    const started = NOW - (index + projectIndex * 3) * 7.6e6;
    const status = STATUSES[(index + projectIndex) % STATUSES.length];
    const family = ["aurora", "cadence", "ember", "lyric"][index % 4];
    return {
      id: `${project.id}-${String(index + 1).padStart(3, "0")}`,
      projectId: project.id,
      name: `${family}-${String(index + 1).padStart(3, "0")}`,
      status,
      startedAt: new Date(started).toISOString(),
      endedAt: status === "running" || status === "queued" ? null : new Date(started + 4.1e6 + index * 1200).toISOString(),
      params: { learning_rate: [0.0001, 0.0002, 0.0003][index % 3], batch_seconds: [120, 180, 240][index % 3], decoder: ["hifigan", "istftnet2", "beetle"][index % 3], seed: 1337 + index, mixed_precision: index % 2 === 0 },
      summary: { "val/mel_loss": 0.42 + ((index * 17) % 31) / 100, "train/generator_total": 1.2 + ((index * 7) % 23) / 10, "system/gpu_utilization_percent": 68 + (index % 27) },
    };
  }),
);

function series(run: Run, name: string, points = 520): MetricPoint[] {
  const offset = Number(run.id.slice(-3)) || 1;
  return Array.from({ length: points }, (_, index) => {
    const progress = index / points;
    let value = 0;
    if (name === "val/mel_loss") value = 1.5 * Math.exp(-progress * 3.2) + 0.31 + Math.sin(index / 21 + offset) * 0.035;
    else if (name === "train/generator_total") value = 4.7 * Math.exp(-progress * 2.4) + 0.9 + Math.sin(index / 13 + offset) * 0.16;
    else if (name === "system/gpu_utilization_percent") value = 78 + Math.sin(index / 9 + offset) * 13 + (offset % 5);
    else value = 0.0003 * (0.5 + Math.cos(index / 80));
    return { runId: run.id, name, step: index * 100, timestamp: Date.parse(run.startedAt) + index * 8000, value: Math.max(0.000001, value) };
  });
}

function artifacts(run: Run): Artifact[] {
  const steps = [4000, 12000, 24000, 40000];
  return steps.flatMap((step, index) => [
    { id: `${run.id}-audio-${step}`, runId: run.id, name: "validation/reference.wav", step, timestamp: Date.parse(run.startedAt) + step * 80, kind: "audio" as const, contentType: "audio/mpeg", sizeBytes: 48320, source: "https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3" },
    { id: `${run.id}-image-${step}`, runId: run.id, name: "validation/mel.png", step, timestamp: Date.parse(run.startedAt) + step * 80, kind: "image" as const, contentType: "image/jpeg", sizeBytes: 138240, source: `https://picsum.photos/seed/${run.id}-${index}/900/420` },
    { id: `${run.id}-text-${step}`, runId: run.id, name: "validation/transcript.txt", step, timestamp: Date.parse(run.startedAt) + step * 80, kind: "text" as const, contentType: "text/plain", sizeBytes: 126, source: `Step ${step}: The model renders stable rhythm and clearer consonants. Run ${run.name} keeps the phrase boundary intact while reducing breath noise.` },
    { id: `${run.id}-plot-${step}`, runId: run.id, name: "validation/attention.plot", step, timestamp: Date.parse(run.startedAt) + step * 80, kind: "plot" as const, contentType: "application/vnd.plotly.v1+json", sizeBytes: 6144, source: JSON.stringify(Array.from({ length: 24 }, (_, bin) => Math.max(4, 82 * Math.exp(-Math.abs(bin - 12 - index) / 5) + ((bin * 17 + index) % 13)))) },
  ]);
}

/** Points behind `metrics(...)`, generated on demand for the requested runs. */
export function pointsFor(runIds: string[], names: string[]): MetricPoint[] {
  return RUNS.filter((run) => runIds.includes(run.id)).flatMap((run) =>
    names.flatMap((name) => series(run, name)),
  );
}

export function runIdsForProject(projectId: string): string[] {
  return RUNS.filter((run) => run.projectId === projectId).map((run) => run.id);
}

export async function listProjects(): Promise<Project[]> { return delay(PROJECTS); }
export async function listRuns(projectId: string): Promise<Run[]> { return delay(RUNS.filter((run) => run.projectId === projectId)); }
export async function getArtifacts(runIds: string[]): Promise<Artifact[]> { return delay(RUNS.filter((run) => runIds.includes(run.id)).flatMap(artifacts)); }

function delay<T>(value: T, ms = 180): Promise<T> { return new Promise((resolve) => window.setTimeout(() => resolve(value), ms)); }
