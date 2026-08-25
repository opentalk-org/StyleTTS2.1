use anyhow::bail;
use sqlx::PgPool;
use tokio::{fs, sync::mpsc};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    db::{fetch_training_samples, fetch_validation_samples},
    loader::{LoadedBatch, LoadedSample, Loader},
    sampling::HistogramSampler,
    server::givemedata,
};

const CACHED_BATCHES: usize = 5;

#[derive(Clone, Copy)]
pub struct Config {
    pub dataset_id: Uuid,
    pub validation_samples: i64,
    pub max_seconds: f32,
    pub max_text_tokens: i32,
    pub seed: u64,
}

pub struct Session {
    pub id: Uuid,
    cancel_token: CancellationToken,
    validation_batches_rx: mpsc::Receiver<anyhow::Result<LoadedBatch>>,
    training_batches_rx: mpsc::Receiver<anyhow::Result<LoadedBatch>>,
}

fn spawn_prefetcher(
    mut sampler: HistogramSampler,
    loader: Loader,
    cancel_token: CancellationToken,
) -> mpsc::Receiver<anyhow::Result<LoadedBatch>> {
    let (tx, rx) = mpsc::channel(CACHED_BATCHES);

    tokio::spawn(async move {
        loop {
            let permit = tokio::select! {
                permit = tx.reserve() => match permit {
                    Ok(permit) => permit,
                    Err(_) => break,
                },
                () = cancel_token.cancelled() => break,
            };

            let loaded = match sampler.next_batch() {
                Ok(batch) => loader.load_batch(batch).await,
                Err(err) => Err(err),
            };
            let failed = loaded.is_err();
            permit.send(loaded);
            if failed {
                break;
            }
        }
    });

    rx
}

async fn read_cached_sample(sample: LoadedSample) -> anyhow::Result<givemedata::Sample> {
    let wave = fs::read(&sample.wave_path).await?.into();
    fs::remove_file(&sample.wave_path).await?;

    Ok(givemedata::Sample {
        wave,
        duration: sample.duration,
        speaker_id: sample.speaker_id,
        language_id: sample.language_id,
        text: sample.text,
    })
}

impl Session {
    pub async fn new(
        pg_pool: &PgPool,
        loader: Loader,
        config: Config,
        plbert_languages: &[String],
    ) -> anyhow::Result<Self> {
        let id = Uuid::new_v4();
        println!("initializing session {id}");

        println!(
            "fetching {} validation rows from dataset {}",
            config.validation_samples, config.dataset_id
        );
        let validation_rows = fetch_validation_samples(pg_pool, config).await?;

        let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

        let validation_sampler =
            HistogramSampler::from_samples(validation_rows, config, plbert_languages)?;

        println!("fetching training rows from dataset {}", config.dataset_id);
        let training_rows = fetch_training_samples(pg_pool, &validation_ids, config).await?;
        let training_sampler =
            HistogramSampler::from_samples(training_rows, config, plbert_languages)?;

        let cancel_token = CancellationToken::new();
        let training_batches_rx =
            spawn_prefetcher(training_sampler, loader.clone(), cancel_token.clone());
        let validation_batches_rx =
            spawn_prefetcher(validation_sampler, loader, cancel_token.clone());

        Ok(Session {
            id,
            cancel_token,
            training_batches_rx,
            validation_batches_rx,
        })
    }

    pub async fn next_batch(
        &mut self,
        validation: bool,
    ) -> anyhow::Result<Vec<givemedata::Sample>> {
        let rx = if validation {
            &mut self.validation_batches_rx
        } else {
            &mut self.training_batches_rx
        };
        let Some(batch) = rx.recv().await else {
            bail!("channel closed, session ended");
        };

        futures::future::try_join_all(batch?.into_iter().map(read_cached_sample)).await
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}
