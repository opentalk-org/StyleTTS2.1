# Metrics viewer feature roadmap

The viewer already covers the basic tracking layer well: run tables, metric charts, smoothing, comparisons, artifacts, checkpoint lineage, model graphs, URL-persisted state, and live ClickHouse updates. The next features should improve analysis rather than add more chart variants.

## Path 1: proven ideas from other tools

| Feature | Inspiration | Value here | Cost |
| --- | --- | --- | --- |
| Group runs and show mean plus variance | W&B, TensorBoard | Very high | Medium |
| Baseline run with metric deltas | W&B | High | Low |
| Parallel coordinates | W&B, ClearML | High | Medium |
| Hyperparameter importance | W&B | High with many runs | Medium |
| Reproducibility comparison | ClearML | Very high | Medium |
| Saved reports and dashboards | W&B | High for collaboration | Medium |
| System-resource dashboard | MLflow, TensorBoard | High | Requires ingestion |
| Synchronized media comparison | W&B, ClearML | High | Low to medium |
| Derived metric formulas | W&B custom charts | High | Medium |
| Alerts and anomaly rules | W&B | Medium | Medium |
| Embedding projector | TensorBoard | Specialized | High |
| Sweep management | W&B, ClearML | Useful, but outside the viewer | High |

### Run grouping and confidence bands

Group runs by parameters such as architecture, learning rate, dataset, or seed. Plot:

- Group mean or median.
- Standard deviation, min/max, or percentile band.
- Number of surviving runs at every step.
- Individual runs faintly behind the aggregate.

This is probably the largest missing everyday feature. W&B exposes parameter-based experiment views and parallel coordinates, while TensorBoard supports multi-run visualization. See [W&B parallel coordinates](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates) and the [TensorBoard overview](https://www.tensorflow.org/tensorboard/get_started).

Lineage needs special handling: aggregate by lineage-relative step, wall time, or steps since the common checkpoint, not blindly by raw step.

### First-class baseline comparison

Let one run or checkpoint be the baseline. Everywhere else show:

- Final-value delta.
- Best-value delta.
- Area-under-curve delta.
- Time-to-threshold improvement.
- Relative percentage improvement.
- Better/worse coloring based on metric direction.

W&B treats baseline runs as a first-class comparison concept and displays summary deltas. See [W&B baseline comparison](https://docs.wandb.ai/models/runs/compare-runs).

### Hyperparameter analysis

Add two panels:

- Parallel coordinates: each run is a line across parameters and outcome metrics.
- Parameter importance: correlations initially, then a tree-based importance model once enough runs exist.

ClearML combines parameter/value comparisons, parallel coordinates, scatter plots, and result metrics; W&B exposes importance and correlation panels. See [ClearML task comparison](https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing/) and [W&B parameter importance](https://docs.wandb.ai/models/app/features/panels/parameter-importance).

Brushing a parameter range should filter the runs table and every metric plot.

### Reproducibility diff

Compare more than metrics:

- Git commit and dirty diff.
- Command and environment.
- Package versions.
- Model architecture.
- Dataset and config identity.
- Input checkpoint.
- Parameter differences.
- Hardware.

ClearML's comparison view highlights differences across source, packages, configuration, models, metrics, plots, and debug samples. See [ClearML comparison](https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing/).

This requires logging reproducibility metadata, but it answers the important question: why did these runs differ?

### Reports instead of only saved views

Turn the current URL view state into a persistent report containing:

- Markdown commentary.
- Frozen or live run selection.
- Metric panels and comparison tables.
- Artifact examples.
- Checkpoint and model links.
- A shareable stable URL.
- An optional snapshot timestamp.

W&B separates disposable workspace exploration from reports containing saved analysis and notes. See [W&B projects and reports](https://docs.wandb.ai/models/track/project-page) and [W&B report panels](https://docs.wandb.ai/models/ref/wandb_workspaces/reports).

### Better sample and media comparison

The current step scrubber could become a comparison matrix:

```text
                 Baseline      Experiment A      Experiment B
sample_001       audio/text    audio/text        audio/text
sample_002       audio/text    audio/text        audio/text
sample_003       audio/text    audio/text        audio/text
```

Lock it by sample identity and checkpoint step, support blind comparison, and show metric deltas beside each sample. W&B supports comparing media across runs, steps, and indices; ClearML compares debug samples by iteration. See [W&B media comparison](https://docs.wandb.ai/models/app/features/panels/media) and [ClearML task comparison](https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing/).

### System metrics and profiler views

Collect and correlate:

- GPU utilization, VRAM, power, and temperature.
- CPU and RAM.
- Disk and network throughput.
- Samples per second and step duration.
- Data-loading time.
- Forward, backward, and optimizer time.

Clicking a loss spike should reveal resource telemetry at the same wall-clock time. MLflow records CPU, GPU, memory, disk, and network metrics; TensorBoard adds profiling-oriented dashboards. See [MLflow system metrics](https://mlflow.org/docs/latest/ml/tracking/system-metrics/) and the [TensorBoard overview](https://www.tensorflow.org/tensorboard/get_started).

## Path 2: lineage-native and model-native ideas

These features exploit data the viewer already understands and could distinguish it from general experiment trackers.

### Fork comparison

When several runs descend from a common checkpoint:

1. Find their nearest common checkpoint.
2. Align curves at the fork.
3. Display only post-fork deltas.
4. Compare improvement per additional step or compute-hour.

The UI could produce a summary such as:

> From checkpoint 42k, branch B gained 0.18 validation score in 7,200 steps; branch A gained 0.11 in 11,000 steps.

This is much more useful than overlaying entire histories.

### Full training replay

One global timeline should control all views simultaneously:

- Metrics.
- Generated samples.
- Histograms.
- Model-graph health.
- Checkpoints.
- Logs and events.
- Resource usage.

Scrubbing to a step reconstructs the observed state of the entire training process. Existing step-linked plots and artifacts provide most of the UI foundation.

### Model health map

Color runtime model-graph nodes using selected-step telemetry:

- Gradient norm.
- Weight norm.
- Update-to-weight ratio.
- Activation mean and variance.
- Zero or saturation percentage.
- NaN and Inf counts.
- Activation or gradient drift.
- Execution time and memory.

Clicking a node should open its history and compare it with the same node in another run or checkpoint. Because the graph comes from actual forward execution, this can handle reused and dynamically invoked modules better than a static layer tree.

### Automatic training-event detection

Detect and annotate:

- Divergence.
- Plateaus.
- Sudden loss changes.
- Gradient explosions.
- Dead layers.
- Throughput degradation.
- Learning-rate transitions.
- Changes immediately following checkpoint restoration.
- Metric changes coinciding with resource pressure.

Events should be stored, searchable, and rendered as vertical chart annotations. Start with deterministic ClickHouse calculations rather than opaque summaries.

### Regression bisector

Given a good descendant and a bad descendant:

- Locate their last common checkpoint.
- Identify the earliest metric divergence.
- Show parameter, configuration, and environment differences.
- Rank model nodes whose activations, gradients, or weights diverged first.
- Identify the first suspicious checkpoint interval.

This turns lineage into an actual debugging tool.

### Checkpoint Pareto frontier

Plot every checkpoint using combinations such as:

- Validation quality versus training cost.
- Quality versus model size.
- Quality versus inference latency.
- Quality versus stability.
- Quality versus elapsed time.

Mark dominated checkpoints and suggest retention candidates. This also enables evidence-based checkpoint pruning.

### Architecture comparison

Match nodes between two runtime model graphs using module path, type, parameter shape, and graph neighborhood. Then show:

- Added and removed operations.
- Shape changes.
- Parameter-count changes.
- Execution-order changes.
- Changed activation or gradient distributions.

This complements an ordinary configuration diff with the model that actually executed.

### Metric relationship explorer

Select a target metric and discover:

- Strong correlations.
- Lagged correlations.
- Metrics that consistently change before failures.
- Relationships that hold only in certain run groups.
- Common change points.

For example:

> Gradient norm increases typically precede validation-loss spikes by 70 to 110 steps in 8 of 10 runs.

ClickHouse is especially suitable for doing this server-side over many runs.

## Recommended order

1. **Baseline comparison and delta columns** — small change and immediately useful.
2. **Grouped curves with percentile bands** — the largest improvement for multi-seed experiments.
3. **Lineage fork comparison** — makes the viewer distinct.
4. **Synchronized training replay** — combines functionality already present.
5. **Model health map** — builds directly on runtime graph and histogram collection.
6. **Parallel coordinates and parameter importance** — valuable once projects contain enough runs.
7. **Persistent reports** — important once multiple people use the viewer.
8. **System and resource telemetry** — excellent, but requires ingestion work.

Sweep orchestration, model deployment, queues, and a full model registry should not be early metrics-viewer priorities. Those are broader platform features and would dilute the strongest opportunity: making lineage, checkpoints, metrics, artifacts, and runtime model behavior explorable as one synchronized training history.
