use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Context;
use bytes::BytesMut;
use clickhouse::Client;
use futures::Stream;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use tokio::sync::mpsc::UnboundedSender;
use tonic::{Status, Streaming};
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::loader::Loader;
use crate::proto::{
    self, AssetResponse, CheckpointRequest, DataRequest, DataResponse, Split, asset_response,
    checkpoint_request,
};
use crate::session::{DataConfig, Session, SessionHandle};
use crate::trainings::{ClaimResult, TrainingStore};

const ASSET_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
pub struct ActiveSession {
    pub handle: SessionHandle,
    pub config: Arc<DataConfig>,
}

pub type SessionsMap = Arc<RwLock<HashMap<Uuid, ActiveSession>>>;

pub fn parse_training_id(value: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(value).map_err(|_| Status::invalid_argument("invalid training ID"))
}

pub struct InitializedTraining {
    pub active: ActiveSession,
    pub train_config: String,
}

pub struct AssetStore {
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
    root: &'static Path,
    synthetic: bool,
}

impl AssetStore {
    pub fn new(
        s3_client: aws_sdk_s3::Client,
        bucket: &'static str,
        root: &'static Path,
        synthetic: bool,
    ) -> Self {
        Self {
            s3_client,
            bucket,
            root,
            synthetic,
        }
    }

    pub async fn ensure(
        &self,
        training_id: Uuid,
        name: &str,
        key: &str,
    ) -> anyhow::Result<PathBuf> {
        let training_dir = self.root.join(training_id.to_string());
        fs::create_dir_all(&training_dir).await?;
        let path = training_dir.join(name);
        if fs::try_exists(&path).await? {
            return Ok(path);
        }

        let part = training_dir.join(format!("{name}.part"));
        info!(training = %training_id, asset = name, key, "downloading asset");
        if self.synthetic {
            fs::write(&part, format!("synthetic asset {name}\n").repeat(1024)).await?;
        } else {
            let mut object = self
                .s3_client
                .get_object()
                .bucket(self.bucket)
                .key(key)
                .send()
                .await
                .with_context(|| format!("fetching asset {name} from {key}"))?;
            let mut file = fs::File::create(&part).await?;
            while let Some(bytes) = object.body.try_next().await? {
                file.write_all(&bytes).await?;
            }
            file.sync_all().await?;
        }
        fs::rename(&part, &path).await?;
        Ok(path)
    }
}

pub async fn initialize(
    training_id: Uuid,
    trainings: &TrainingStore,
    assets: &AssetStore,
    database: &Client,
    loader: Arc<dyn Loader>,
    cache_dir: &'static Path,
    synthetic: bool,
) -> Result<InitializedTraining, Status> {
    let claimed = trainings.claim(training_id).await.map_err(|err| {
        error!(training = %training_id, error = format!("{err:#}"), "claiming training failed");
        Status::internal(format!("{err:#}"))
    })?;
    let (data_config, train_config) = match claimed {
        ClaimResult::Claimed(v1, v2) => (v1, v2),
        ClaimResult::NotFound => return Err(Status::not_found("unknown training")),
        ClaimResult::Unavailable(state) => {
            return Err(Status::failed_precondition(format!(
                "training is {}",
                state.as_str()
            )));
        }
    };
    let config = Arc::new(data_config);
    let initialized: anyhow::Result<SessionHandle> = async {
        futures::future::try_join_all(
            config
                .assets
                .iter()
                .map(|(name, asset)| assets.ensure(training_id, name, &asset.object)),
        )
        .await?;
        let session =
            Session::new(training_id, database, loader, cache_dir, &config, synthetic).await?;
        Ok(SessionHandle::spawn(session))
    }
    .await;
    match initialized {
        Ok(handle) => Ok(InitializedTraining {
            active: ActiveSession { handle, config },
            train_config: train_config,
        }),
        Err(err) => {
            error!(training = %training_id, error = format!("{err:#}"), "training initialization failed");
            if let Err(reset_err) = trainings.reset(training_id).await {
                error!(training = %training_id, error = format!("{reset_err:#}"), "resetting training state failed");
            }
            Err(Status::internal(format!("{err:#}")))
        }
    }
}

pub async fn data_handler(
    sessions: SessionsMap,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        let training_id = parse_training_id(&req.training_id)?;
        debug!(training = %training_id, split = ?req.split(), "data request");
        let active = sessions
            .read()
            .await
            .get(&training_id)
            .cloned()
            .context("unknown training")?;
        let Some(loaded_batch) = active
            .handle
            .next_batch(req.split() == Split::Validation)
            .await?
        else {
            debug!(training = %training_id, split = ?req.split(), "data stream exhausted");
            return Ok(());
        };
        let batch = loaded_batch.into_iter().map(proto::Sample::from).collect();
        resp_stream.send(Ok(DataResponse { batch }))?;
    }
    Ok(())
}

pub fn asset_stream(
    path: PathBuf,
    entrypoint: Option<String>,
) -> Pin<Box<dyn Stream<Item = Result<AssetResponse, Status>> + Send>> {
    Box::pin(async_stream::stream! {
        yield Ok(AssetResponse {
            payload: Some(asset_response::Payload::Metadata(proto::AssetMetadata {
                entrypoint,
            })),
        });
        let mut file = match fs::File::open(&path).await {
            Ok(file) => file,
            Err(err) => {
                yield Err(Status::internal(format!("{err:#}")));
                return;
            }
        };
        loop {
            let mut buf = BytesMut::with_capacity(ASSET_CHUNK_BYTES);
            match file.read_buf(&mut buf).await {
                Ok(0) => break,
                Ok(_) => yield Ok(AssetResponse {
                    payload: Some(asset_response::Payload::Chunk(buf.freeze())),
                }),
                Err(err) => {
                    yield Err(Status::internal(format!("{err:#}")));
                    break;
                }
            }
        }
    })
}

pub async fn receive_checkpoint(
    root: &Path,
    training_id: Uuid,
    step: u64,
    stream: &mut Streaming<CheckpointRequest>,
) -> anyhow::Result<(PathBuf, u64)> {
    let dir = root.join(training_id.to_string());
    fs::create_dir_all(&dir).await?;
    let path = dir.join(format!("step_{step:09}.tar"));
    let part = dir.join(format!("step_{step:09}.tar.part"));
    let mut file = fs::File::create(&part).await?;
    let mut bytes = 0;
    while let Some(request) = stream.message().await? {
        match request.payload {
            Some(checkpoint_request::Payload::Chunk(chunk)) => {
                bytes += chunk.len() as u64;
                file.write_all(&chunk).await?;
            }
            _ => anyhow::bail!("expected checkpoint chunks after the metadata"),
        }
    }
    file.sync_all().await?;
    fs::rename(&part, &path).await?;
    Ok((path, bytes))
}
