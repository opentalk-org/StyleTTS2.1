use std::{path::Path, pin::Pin, sync::Arc};

use anyhow::bail;
use futures::{Stream, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, debug, info, info_span};
use uuid::Uuid;

use crate::{
    db::{
        TRAINING_SEED_SALT, VALIDATION_SEED_SALT, fetch_training_samples, fetch_validation_samples,
        synthetic_rows,
    },
    loader::Loader,
    prefetch::{LoadedBatch, LoadedSample, Prefetcher},
    sampling::HistogramSampler,
};

const SYNTHETIC_TRAINING_SAMPLES: usize = 256;

#[derive(Clone, Copy)]
pub struct Config {
    pub dataset_id: Uuid,
    pub validation_samples: i64,
    pub max_seconds: f32,
    pub max_text_tokens: i32,
    pub seed: u64,
    pub synthetic: bool,
}

enum Batches {
    Prefetched(Prefetcher),
    OnDemand(Pin<Box<dyn Stream<Item = anyhow::Result<LoadedBatch>> + Send>>),
}

fn on_demand_batches(mut sampler: HistogramSampler, loader: Arc<dyn Loader>) -> Batches {
    Batches::OnDemand(Box::pin(async_stream::stream! {
        loop {
            let batch = match sampler.next_batch() {
                Ok(batch) => {
                    let batch = loader.load_batch( batch).await?;
                    Ok(batch.into_iter().map(LoadedSample::from).collect())
                }
                Err(err) => Err(err),
            };
            yield batch;
        }
    }))
}

pub struct Session {
    pub id: Uuid,
    cancel_token: CancellationToken,
    validation_batches: Batches,
    training_batches: Batches,
}

impl Session {
    pub async fn new(
        pg_pool: &sqlx::PgPool,
        loader: Arc<dyn Loader>,
        cache_dir: &'static Path,
        config: Config,
        plbert_languages: &[String],
    ) -> anyhow::Result<Self> {
        let id = Uuid::new_v4();
        info!(session = %id, dataset = %config.dataset_id, synthetic = config.synthetic, "initializing session");

        let (validation_rows, training_rows) = if config.synthetic {
            let language = plbert_languages.first().map(String::as_str).unwrap_or("en");
            let validation_rows = synthetic_rows(
                config,
                language,
                config.validation_samples as usize,
                VALIDATION_SEED_SALT,
            );
            let training_rows = synthetic_rows(
                config,
                language,
                SYNTHETIC_TRAINING_SAMPLES,
                TRAINING_SEED_SALT,
            );
            info!(
                validation = validation_rows.len(),
                training = training_rows.len(),
                "generated synthetic rows"
            );
            (validation_rows, training_rows)
        } else {
            let validation_rows = fetch_validation_samples(pg_pool, config).await?;
            info!(
                rows = validation_rows.len(),
                requested = config.validation_samples,
                "fetched validation rows"
            );

            let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

            let training_rows = fetch_training_samples(pg_pool, &validation_ids, config).await?;
            info!(rows = training_rows.len(), "fetched training rows");
            (validation_rows, training_rows)
        };

        let validation_sampler =
            HistogramSampler::from_samples(validation_rows, config, plbert_languages)?;
        let training_sampler =
            HistogramSampler::from_samples(training_rows, config, plbert_languages)?;

        let cancel_token = CancellationToken::new();
        let (training_batches, validation_batches) = if config.synthetic {
            (
                on_demand_batches(training_sampler, loader.clone()),
                on_demand_batches(validation_sampler, loader),
            )
        } else {
            (
                Batches::Prefetched(Prefetcher::spawn(
                    training_sampler,
                    loader.clone(),
                    cache_dir,
                    cancel_token.clone(),
                    info_span!("prefetcher", session = %id, split = "training"),
                )),
                Batches::Prefetched(Prefetcher::spawn(
                    validation_sampler,
                    loader,
                    cache_dir,
                    cancel_token.clone(),
                    info_span!("prefetcher", session = %id, split = "validation"),
                )),
            )
        };

        Ok(Session {
            id,
            cancel_token,
            training_batches,
            validation_batches,
        })
    }

    pub async fn next_batch(&mut self, validation: bool) -> anyhow::Result<LoadedBatch> {
        let batches = if validation {
            &mut self.validation_batches
        } else {
            &mut self.training_batches
        };
        match batches {
            Batches::Prefetched(prefetcher) => prefetcher.next_batch().await,
            Batches::OnDemand(stream) => match stream.next().await {
                Some(batch) => batch,
                None => bail!("batch stream ended"),
            },
        }
    }

    pub async fn finish(self) {
        self.cancel_token.cancel();
        let prefetchers = match (self.validation_batches, self.training_batches) {
            (Batches::Prefetched(p1), Batches::Prefetched(p2)) => vec![p1, p2],
            (Batches::Prefetched(p), _) | (_, Batches::Prefetched(p)) => vec![p],
            _ => return,
        };
        futures::future::join_all(prefetchers.into_iter().map(Prefetcher::drain)).await;
    }
}

enum Command {
    NextBatch {
        validation: bool,
        reply: oneshot::Sender<anyhow::Result<LoadedBatch>>,
    },
    Finish {
        reply: oneshot::Sender<()>,
    },
}

// The actor task owns the Session by value: batch requests are serialized through the
// channel, and consuming `Session::finish` runs on every exit path.
#[derive(Clone)]
pub struct SessionHandle {
    pub id: Uuid,
    tx: mpsc::Sender<Command>,
}

impl SessionHandle {
    pub fn spawn(session: Session) -> Self {
        let id = session.id;
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(run(session, rx).instrument(info_span!("session", session = %id)));
        Self { id, tx }
    }

    pub async fn next_batch(&self, validation: bool) -> anyhow::Result<LoadedBatch> {
        let (reply, response) = oneshot::channel();
        if self
            .tx
            .send(Command::NextBatch { validation, reply })
            .await
            .is_err()
        {
            bail!("session ended");
        }
        match response.await {
            Ok(batch) => batch,
            Err(_) => bail!("session ended"),
        }
    }

    pub async fn finish(&self) {
        let (reply, response) = oneshot::channel();
        if self.tx.send(Command::Finish { reply }).await.is_ok() {
            // wait for the drain to complete; an error means the actor is already gone
            let _ = response.await;
        }
    }
}

async fn run(mut session: Session, mut rx: mpsc::Receiver<Command>) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Command::NextBatch { validation, reply } => {
                let _ = reply.send(session.next_batch(validation).await);
            }
            Command::Finish { reply } => {
                session.finish().await;
                let _ = reply.send(());
                debug!("session finished");
                return;
            }
        }
    }

    // all handles dropped without an End: still stop prefetchers and drain the cache
    session.finish().await;
    debug!("session finished after handles dropped");
}
