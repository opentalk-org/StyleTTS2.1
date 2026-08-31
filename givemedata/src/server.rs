use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use crate::loader::{Loader, S3Loader, SyntheticLoader};
use crate::metrics;
use crate::server::givemedata::{
    AssetRequest, AssetResponse, CheckpointRequest, CheckpointResponse, EndRequest, EndResponse,
    InitRequest, InitResponse, MetricsRequest, MetricsResponse, asset_response, checkpoint_request,
    metrics_request,
};
use crate::session::{DataConfig, Session, SessionHandle};
use anyhow::Context;
use bytes::BytesMut;
use futures::Stream;
use sqlx::{PgPool, Pool, Postgres};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{RwLock, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};

pub mod givemedata {
    tonic::include_proto!("_");
}
use givemedata::give_me_data_server::{GiveMeData as GiveMeDataService, GiveMeDataServer};
use givemedata::{DataRequest, DataResponse, Split};

type SessionsMap = Arc<RwLock<HashMap<String, SessionHandle>>>;

const ASSET_CHUNK_BYTES: usize = 2 * 1024 * 1024;

struct GiveMeData {
    db_pool: PgPool,
    loader: Arc<dyn Loader>,
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
    cache_dir: &'static Path,
    assets_dir: &'static Path,
    checkpoint_dir: &'static Path,
    metrics_dir: &'static Path,
    synthetic: bool,
    data_config: &'static DataConfig,
    train_config: &'static str,
    sessions: SessionsMap,
}

impl GiveMeData {
    async fn new(
        s3_client: aws_sdk_s3::Client,
        pg_pool: Pool<Postgres>,
        bucket: &'static str,
        cache_dir: &'static Path,
        assets_dir: &'static Path,
        checkpoint_dir: &'static Path,
        metrics_dir: &'static Path,
        synthetic: bool,
        data_config: &'static DataConfig,
        train_config: &'static str,
    ) -> Result<Self, sqlx::Error> {
        let loader: Arc<dyn Loader> = if synthetic {
            Arc::new(SyntheticLoader)
        } else {
            Arc::new(S3Loader::new(s3_client.clone(), bucket))
        };
        Ok(GiveMeData {
            sessions: Default::default(),
            loader,
            s3_client,
            bucket,
            cache_dir,
            assets_dir,
            checkpoint_dir,
            metrics_dir,
            synthetic,
            data_config,
            train_config,
            db_pool: pg_pool,
        })
    }

    /// Downloads the asset into the assets dir unless it is already there.
    /// Writes to a .part file first so a crashed download is never served.
    async fn ensure_asset(&self, name: &str, key: &str) -> anyhow::Result<PathBuf> {
        let path = self.assets_dir.join(name);
        if fs::try_exists(&path).await? {
            return Ok(path);
        }

        let part = self.assets_dir.join(format!("{name}.part"));
        info!(asset = name, key, "downloading asset");
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

async fn data_handler(
    sessions: SessionsMap,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        debug!(session = %req.session_id, split = ?req.split(), "data request");
        let session = sessions
            .read()
            .await
            .get(&req.session_id)
            .cloned()
            .context("unknown session")?;

        let Some(loaded_batch) = session.next_batch(req.split() == Split::Validation).await? else {
            debug!(
                session = %req.session_id,
                split = ?req.split(),
                "data stream exhausted"
            );
            return Ok(());
        };
        let batch = loaded_batch
            .into_iter()
            .map(|sample| givemedata::Sample {
                wave: sample.wave,
                duration: sample.duration,
                speaker_id: sample.speaker_id,
                language_id: sample.language_id,
                text: sample.text,
            })
            .collect();
        resp_stream.send(Ok(DataResponse { batch }))?;
    }

    Ok(())
}

#[tonic::async_trait]
impl GiveMeDataService for GiveMeData {
    async fn init(&self, _request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        debug!("init request");

        futures::future::try_join_all(
            self.data_config
                .assets
                .iter()
                .map(|(name, asset)| self.ensure_asset(name, &asset.object)),
        )
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "asset prefetch failed");
            Status::internal(format!("{err:#}"))
        })?;

        let session = Session::new(
            &self.db_pool,
            self.loader.clone(),
            self.cache_dir,
            self.data_config,
            self.synthetic,
        )
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "session init failed");
            Status::internal(format!("{err:#}"))
        })?;
        let handle = SessionHandle::spawn(session);
        let session_id = handle.id.to_string();

        self.sessions
            .write()
            .await
            .insert(session_id.clone(), handle);
        info!(session = %session_id, "session created");

        Ok(Response::new(InitResponse {
            session_id,
            train_config: self.train_config.to_string(),
        }))
    }

    type DataStream = UnboundedReceiverStream<Result<DataResponse, Status>>;

    async fn data(
        &self,
        request: Request<Streaming<DataRequest>>,
    ) -> Result<Response<Self::DataStream>, Status> {
        let mut stream = request.into_inner();

        let (out_tx, out_rx) = mpsc::unbounded_channel();
        tokio::spawn({
            let sessions = self.sessions.clone();
            async move {
                if let Err(err) = data_handler(sessions, &mut stream, &out_tx).await {
                    error!(error = format!("{err:#}"), "data stream failed");
                    let _ = out_tx.send(Err(Status::internal(format!("{err:#}"))));
                }
            }
        });

        Ok(UnboundedReceiverStream::new(out_rx).into())
    }

    type AssetStream = Pin<Box<dyn Stream<Item = Result<AssetResponse, Status>> + Send>>;

    async fn asset(
        &self,
        request: Request<AssetRequest>,
    ) -> Result<Response<Self::AssetStream>, Status> {
        let request = request.into_inner();
        if !self.sessions.read().await.contains_key(&request.session_id) {
            return Err(Status::not_found("unknown session"));
        }
        let asset = self
            .data_config
            .assets
            .get(&request.name)
            .ok_or_else(|| Status::not_found(format!("unknown asset {:?}", request.name)))?;
        info!(session = %request.session_id, asset = %request.name, "asset requested");

        let path = self
            .ensure_asset(&request.name, &asset.object)
            .await
            .map_err(|err| {
                error!(error = format!("{err:#}"), asset = %request.name, "asset fetch failed");
                Status::internal(format!("{err:#}"))
            })?;
        let entrypoint = asset.entrypoint.clone();

        let stream = async_stream::stream! {
            yield Ok(AssetResponse {
                payload: Some(asset_response::Payload::Metadata(givemedata::AssetMetadata {
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
        };

        Ok(Response::new(Box::pin(stream)))
    }

    async fn checkpoint(
        &self,
        request: Request<Streaming<CheckpointRequest>>,
    ) -> Result<Response<CheckpointResponse>, Status> {
        let mut stream = request.into_inner();

        let metadata = match stream.message().await?.and_then(|r| r.payload) {
            Some(checkpoint_request::Payload::Metadata(metadata)) => metadata,
            _ => {
                return Err(Status::invalid_argument(
                    "first checkpoint message must be metadata",
                ));
            }
        };
        if !self
            .sessions
            .read()
            .await
            .contains_key(&metadata.session_id)
        {
            return Err(Status::not_found("unknown session"));
        }
        info!(session = %metadata.session_id, step = metadata.step, "receiving checkpoint");

        let result: anyhow::Result<(PathBuf, u64)> = async {
            let dir = self.checkpoint_dir.join(&metadata.session_id);
            fs::create_dir_all(&dir).await?;
            let path = dir.join(format!("step_{:09}.tar", metadata.step));
            let part = dir.join(format!("step_{:09}.tar.part", metadata.step));

            let mut file = fs::File::create(&part).await?;
            let mut bytes: u64 = 0;
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
        .await;

        match result {
            Ok((path, bytes)) => {
                info!(
                    session = %metadata.session_id,
                    step = metadata.step,
                    bytes,
                    path = %path.display(),
                    "checkpoint stored"
                );
                Ok(Response::new(CheckpointResponse {}))
            }
            Err(err) => {
                error!(error = format!("{err:#}"), "storing checkpoint failed");
                Err(Status::internal(format!("{err:#}")))
            }
        }
    }

    async fn metrics(
        &self,
        request: Request<Streaming<MetricsRequest>>,
    ) -> Result<Response<MetricsResponse>, Status> {
        let mut stream = request.into_inner();
        let metadata = match stream.message().await?.and_then(|request| request.payload) {
            Some(metrics_request::Payload::Metadata(metadata)) => metadata,
            _ => {
                return Err(Status::invalid_argument(
                    "first metrics message must be stream metadata",
                ));
            }
        };
        if !self
            .sessions
            .read()
            .await
            .contains_key(&metadata.session_id)
        {
            return Err(Status::not_found("unknown session"));
        }
        info!(session = %metadata.session_id, "receiving metrics");

        match metrics::receive(self.metrics_dir, &metadata.session_id, stream).await {
            Ok(response) => {
                info!(
                    session = %metadata.session_id,
                    metrics = response.metrics_received,
                    artifacts = response.artifacts_received,
                    artifact_bytes = response.artifact_bytes_received,
                    "metrics stored"
                );
                Ok(Response::new(response))
            }
            Err(status) => {
                error!(
                    session = %metadata.session_id,
                    error = %status,
                    "storing metrics failed"
                );
                Err(status)
            }
        }
    }

    async fn end(&self, request: Request<EndRequest>) -> Result<Response<EndResponse>, Status> {
        let request = request.into_inner();
        info!(session = %request.session_id, "ending session");
        let removed = self.sessions.write().await.remove(&request.session_id);

        match removed {
            None => return Err(Status::not_found("unknown session")),
            Some(handle) => handle.finish().await,
        }

        Ok(Response::new(EndResponse {}))
    }
}

pub async fn serve(
    port: u16,
    s3_client: aws_sdk_s3::Client,
    pg_pool: Pool<Postgres>,
    bucket: &'static str,
    cache_dir: &'static Path,
    assets_dir: &'static Path,
    checkpoint_dir: &'static Path,
    metrics_dir: &'static Path,
    synthetic: bool,
    data_config: &'static DataConfig,
    train_config: &'static str,
) -> anyhow::Result<()> {
    if synthetic {
        info!("serving synthetic sessions");
    }
    info!("listening on 0.0.0.0:{port}");

    Server::builder()
        .add_service(GiveMeDataServer::new(
            GiveMeData::new(
                s3_client,
                pg_pool,
                bucket,
                cache_dir,
                assets_dir,
                checkpoint_dir,
                metrics_dir,
                synthetic,
                data_config,
                train_config,
            )
            .await?,
        ))
        .serve(SocketAddr::from((Ipv4Addr::new(0, 0, 0, 0), port)))
        .await?;

    Ok(())
}
