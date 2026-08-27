use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;

use crate::loader::{Loader, S3Loader, SyntheticLoader};
use crate::server::givemedata::{EndRequest, EndResponse, InitRequest, InitResponse};
use crate::session::{self, Session};
use anyhow::Context;
use sqlx::{PgPool, Pool, Postgres};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, error, info};
use uuid::Uuid;

pub mod givemedata {
    tonic::include_proto!("_");
}
use givemedata::give_me_data_server::{GiveMeData as GiveMeDataService, GiveMeDataServer};
use givemedata::{DataRequest, DataResponse, Split};

type SessionsMap = Arc<RwLock<HashMap<String, Arc<Mutex<Session>>>>>;

struct GiveMeData {
    db_pool: PgPool,
    loader: Arc<dyn Loader>,
    cache_dir: &'static Path,
    synthetic: bool,
    sessions: SessionsMap,
}

impl GiveMeData {
    async fn new(
        s3_client: aws_sdk_s3::Client,
        pg_pool: Pool<Postgres>,
        bucket: &'static str,
        cache_dir: &'static Path,
        synthetic: bool,
    ) -> Result<Self, sqlx::Error> {
        let loader: Arc<dyn Loader> = if synthetic {
            Arc::new(SyntheticLoader)
        } else {
            Arc::new(S3Loader::new(s3_client, bucket))
        };
        Ok(GiveMeData {
            sessions: Default::default(),
            loader,
            cache_dir,
            synthetic,
            db_pool: pg_pool,
        })
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
        let mut session = session.lock().await;

        let loaded_batch = session.next_batch(req.split() == Split::Validation).await?;
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
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        let request = request.into_inner();
        debug!(?request, "init request");

        let dataset_id = Uuid::try_parse(&request.dataset_id).map_err(|e| {
            Status::invalid_argument(format!("Could not parse dataset_id UUID: {e}"))
        })?;

        let config = session::Config {
            dataset_id,
            validation_samples: request.validation_samples as i64,
            max_seconds: request.max_seconds,
            max_text_tokens: request.max_text_tokens,
            seed: request.seed,
            synthetic: self.synthetic,
        };

        let session = Session::new(
            &self.db_pool,
            self.loader.clone(),
            self.cache_dir,
            config,
            &request.plbert_languages,
        )
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "session init failed");
            Status::internal(format!("{err:#}"))
        })?;
        let session_id = session.id.to_string();

        self.sessions
            .write()
            .await
            .insert(session_id.clone(), Arc::new(Mutex::new(session)));
        info!(session = %session_id, "session created");

        Ok(Response::new(InitResponse { session_id }))
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

    async fn end(&self, request: Request<EndRequest>) -> Result<Response<EndResponse>, Status> {
        let request = request.into_inner();
        info!(session = %request.session_id, "ending session");
        let mut sessions = self.sessions.write().await;
        let removed = sessions.remove(&request.session_id);

        if removed.is_none() {
            return Err(Status::internal("unknown session"));
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
    synthetic: bool,
) -> anyhow::Result<()> {
    if synthetic {
        info!("serving synthetic sessions");
    }
    info!("listening on 0.0.0.0:{port}");

    Server::builder()
        .add_service(GiveMeDataServer::new(
            GiveMeData::new(s3_client, pg_pool, bucket, cache_dir, synthetic).await?,
        ))
        .serve(SocketAddr::from((Ipv4Addr::new(0, 0, 0, 0), port)))
        .await?;

    Ok(())
}
