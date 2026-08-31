use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tonic::{Status, Streaming};
use tracing::warn;

use crate::server::givemedata::{
    ArtifactMetric, MetricsRequest, MetricsResponse, ScalarMetric, metrics_request,
};

#[derive(Serialize)]
struct ScalarRecord<'a> {
    step: u64,
    timestamp_unix_ms: i64,
    name: &'a str,
    value: f64,
}

#[derive(Serialize)]
struct ArtifactRecord<'a> {
    step: u64,
    timestamp_unix_ms: i64,
    name: &'a str,
    content_type: &'a str,
    size_bytes: u64,
}

struct PendingArtifact {
    metadata: ArtifactMetric,
    target: PathBuf,
    part: PathBuf,
    file: File,
    received: u64,
}

pub async fn receive(
    root: &Path,
    session_id: &str,
    mut stream: Streaming<MetricsRequest>,
) -> Result<MetricsResponse, Status> {
    let session_dir = root.join(session_id);
    fs::create_dir_all(&session_dir).await.map_err(internal)?;
    let metrics_path = session_dir.join("metrics.jsonl");
    let artifacts_path = session_dir.join("artifacts.jsonl");
    let artifacts_dir = session_dir.join("artifacts");
    fs::create_dir_all(&artifacts_dir).await.map_err(internal)?;

    let mut pending: Option<PendingArtifact> = None;
    let mut response = MetricsResponse::default();
    let result = async {
        while let Some(request) = stream.message().await? {
            match request.payload {
                Some(metrics_request::Payload::Metadata(_)) => {
                    return Err(Status::invalid_argument(
                        "metrics stream metadata may only appear first",
                    ));
                }
                Some(metrics_request::Payload::Metric(metric)) => {
                    store_metric(&metrics_path, &metric).await?;
                    response.metrics_received += 1;
                }
                Some(metrics_request::Payload::Artifact(artifact)) => {
                    if let Some(previous) = pending.take() {
                        warn!(
                            artifact = %previous.metadata.name,
                            received = previous.received,
                            expected = previous.metadata.size_bytes,
                            "artifact upload cancelled by a new artifact"
                        );
                        discard_artifact(previous).await?;
                    }
                    let artifact = begin_artifact(&artifacts_dir, artifact).await?;
                    if artifact.metadata.size_bytes == 0 {
                        finish_artifact(artifact, &artifacts_path).await?;
                        response.artifacts_received += 1;
                    } else {
                        pending = Some(artifact);
                    }
                }
                Some(metrics_request::Payload::ArtifactChunk(chunk)) => {
                    let Some(mut artifact) = pending.take() else {
                        return Err(Status::invalid_argument(
                            "artifact chunk has no artifact metadata",
                        ));
                    };
                    let chunk_size = chunk.data.len() as u64;
                    if artifact.received + chunk_size > artifact.metadata.size_bytes {
                        pending = Some(artifact);
                        return Err(Status::invalid_argument(
                            "artifact contains more bytes than declared",
                        ));
                    }
                    if let Err(error) = artifact.file.write_all(&chunk.data).await {
                        pending = Some(artifact);
                        return Err(internal(error));
                    }
                    artifact.received += chunk_size;
                    response.artifact_bytes_received += chunk_size;
                    if artifact.received == artifact.metadata.size_bytes {
                        finish_artifact(artifact, &artifacts_path).await?;
                        response.artifacts_received += 1;
                    } else {
                        pending = Some(artifact);
                    }
                }
                None => return Err(Status::invalid_argument("metrics message has no payload")),
            }
        }

        if pending.is_some() {
            return Err(Status::invalid_argument(
                "metrics stream ended before the artifact was complete",
            ));
        }
        Ok(())
    }
    .await;

    if result.is_err() {
        if let Some(artifact) = pending {
            let _ = discard_artifact(artifact).await;
        }
    }
    result?;
    Ok(response)
}

async fn store_metric(path: &Path, metric: &ScalarMetric) -> Result<(), Status> {
    if metric.name.is_empty() {
        return Err(Status::invalid_argument("metric name cannot be empty"));
    }
    append_json(
        path,
        &ScalarRecord {
            step: metric.step,
            timestamp_unix_ms: metric.timestamp_unix_ms,
            name: &metric.name,
            value: metric.value,
        },
    )
    .await
}

async fn begin_artifact(
    artifacts_dir: &Path,
    metadata: ArtifactMetric,
) -> Result<PendingArtifact, Status> {
    let relative = artifact_path(&metadata.name)?;
    let target = artifacts_dir.join(relative);
    let parent = target
        .parent()
        .ok_or_else(|| Status::invalid_argument("artifact name has no parent directory"))?;
    fs::create_dir_all(parent).await.map_err(internal)?;
    let filename = target
        .file_name()
        .ok_or_else(|| Status::invalid_argument("artifact name has no filename"))?
        .to_string_lossy();
    let part = target.with_file_name(format!("{filename}.part"));
    let file = File::create(&part).await.map_err(internal)?;
    Ok(PendingArtifact {
        metadata,
        target,
        part,
        file,
        received: 0,
    })
}

async fn finish_artifact(artifact: PendingArtifact, artifacts_path: &Path) -> Result<(), Status> {
    artifact.file.sync_all().await.map_err(internal)?;
    drop(artifact.file);
    if fs::try_exists(&artifact.target).await.map_err(internal)? {
        fs::remove_file(&artifact.target).await.map_err(internal)?;
    }
    fs::rename(&artifact.part, &artifact.target)
        .await
        .map_err(internal)?;
    append_json(
        artifacts_path,
        &ArtifactRecord {
            step: artifact.metadata.step,
            timestamp_unix_ms: artifact.metadata.timestamp_unix_ms,
            name: &artifact.metadata.name,
            content_type: &artifact.metadata.content_type,
            size_bytes: artifact.metadata.size_bytes,
        },
    )
    .await
}

async fn discard_artifact(artifact: PendingArtifact) -> Result<(), Status> {
    drop(artifact.file);
    fs::remove_file(artifact.part).await.map_err(internal)
}

async fn append_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Status> {
    let mut line = serde_json::to_vec(value).map_err(internal)?;
    line.push(b'\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(internal)?;
    file.write_all(&line).await.map_err(internal)?;
    file.flush().await.map_err(internal)
}

fn artifact_path(name: &str) -> Result<&Path, Status> {
    if name.is_empty() {
        return Err(Status::invalid_argument("artifact name cannot be empty"));
    }
    let path = Path::new(name);
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Status::invalid_argument(
            "artifact name must be a relative path without parent components",
        ));
    }
    Ok(path)
}

fn internal(error: impl std::fmt::Display) -> Status {
    Status::internal(format!("{error:#}"))
}
