use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use crate::loader::Loader;
use crate::server::givemedata::{InitRequest, InitResponse};
use crate::session::Session;
use anyhow::Context;
use sqlx::{PgPool, Pool, Postgres};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use uuid::Uuid;

pub mod givemedata {
    tonic::include_proto!("_");
}
use givemedata::give_me_data_server::{GiveMeData as GiveMeDataService, GiveMeDataServer};
use givemedata::{DataRequest, DataResponse, Split};

type SessionsMap = Arc<RwLock<HashMap<String, Arc<Mutex<Session>>>>>;

struct GiveMeData {
    db_pool: PgPool,
    loader: Loader,
    sessions: SessionsMap,
}

impl GiveMeData {
    async fn new(
        s3_client: aws_sdk_s3::Client,
        pg_pool: Pool<Postgres>,
        bucket: String,
    ) -> Result<Self, sqlx::Error> {
        Ok(GiveMeData {
            sessions: Default::default(),
            loader: Loader::new(s3_client, bucket),
            db_pool: pg_pool,
        })
    }
}

async fn data_handler(
    sessions: SessionsMap,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
    loader: &Loader,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        println!("{req:?}");
        println!("locking session {}", req.session_id);
        let session = sessions
            .read()
            .await
            .get(&req.session_id)
            .cloned()
            .context("unknown session")?;
        let mut session = session.lock().await;

        if req.split() == Split::Validation {
            let batch = session.validation_histogram.next_batch()?;

            println!("loading batch of {}", batch.len());
            let mut loaded_batch: Vec<givemedata::Sample> = vec![];

            let mut set = tokio::task::JoinSet::new();
            for sample in batch {
                let loader = loader.clone();
                set.spawn(async move { loader.load_sample(sample).await });
            }
            for res in set.join_all().await {
                loaded_batch.push(res?);
            }

            resp_stream.send(Ok(DataResponse {
                batch: loaded_batch,
            }))?;
        } else {
            // yield traiinig
            resp_stream.send(Ok(DataResponse { batch: vec![] }))?;
        }
    }

    Ok(())
}

#[tonic::async_trait]
impl GiveMeDataService for GiveMeData {
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status> {
        // preprocessing, prefetching, other shit like that
        let request = request.into_inner();
        println!("{request:?}");

        let dataset_id = Uuid::try_parse(&request.dataset_id).map_err(|e| {
            Status::invalid_argument(format!("Could not parse dataset_id UUID: {e}"))
        })?;

        let session = Session::new(
            &self.db_pool,
            dataset_id,
            request.validation_samples as i64,
            request.max_seconds,
            request.max_text_tokens,
            &request.plbert_languages,
        )
        .await
        .map_err(|e| Status::internal(e.to_string()))?;
        let session_id = session.id.to_string();

        self.sessions
            .write()
            .await
            .insert(session_id.clone(), Arc::new(Mutex::new(session)));

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
            let loader = self.loader.clone();
            async move {
                if let Err(err) = data_handler(sessions, &mut stream, &out_tx, &loader).await {
                    println!("erra: {err}");
                }
            }
        });

        Ok(UnboundedReceiverStream::new(out_rx).into())
    }
}

pub async fn serve(
    port: u16,
    s3_client: aws_sdk_s3::Client,
    pg_pool: Pool<Postgres>,
    bucket: String,
) -> anyhow::Result<()> {
    println!("[givemedata] listening on 0.0.0.0:{}", port);

    Server::builder()
        .add_service(GiveMeDataServer::new(
            GiveMeData::new(s3_client, pg_pool, bucket).await?,
        ))
        .serve(SocketAddr::from((Ipv4Addr::new(0, 0, 0, 0), port)))
        .await?;

    Ok(())
}
