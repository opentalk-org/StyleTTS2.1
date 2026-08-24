use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    db::{fetch_training_samples, fetch_validation_samples},
    sampling::HistogramSampler,
};

#[derive(Clone, Copy)]
pub struct Config {
    pub dataset_id: Uuid,
    pub validation_samples: i64,
    pub max_seconds: f32,
    pub max_text_tokens: i32,
    pub seed: u64,
    pub plbert_languages: &'static [String],
}

pub struct Session {
    pub validation_sampler: HistogramSampler,
    pub training_sampler: HistogramSampler,
    pub id: Uuid,
}

impl Session {
    pub async fn new(pg_pool: &PgPool, config: Config) -> anyhow::Result<Self> {
        let id = Uuid::new_v4();
        println!("initializing session {id}");

        println!(
            "fetching {} validation rows from dataset {}",
            config.validation_samples, config.dataset_id
        );
        let validation_rows = fetch_validation_samples(pg_pool, config).await?;

        let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

        let validation_histogram = HistogramSampler::from_samples(validation_rows, config)?;

        println!("fetching training rows from dataset {}", config.dataset_id);
        let training_rows = fetch_training_samples(pg_pool, &validation_ids, config).await?;
        let training_histogram = HistogramSampler::from_samples(training_rows, config)?;

        Ok(Session {
            validation_sampler: validation_histogram,
            training_sampler: training_histogram,
            id,
        })
    }
}
