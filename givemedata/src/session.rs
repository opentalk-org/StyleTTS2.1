use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    db::{fetch_training_samples, fetch_validation_samples},
    sampling::HistogramSampler,
};

pub struct Session {
    pub validation_sampler: HistogramSampler,
    pub training_sampler: HistogramSampler,
    pub id: Uuid,
}

impl Session {
    pub async fn new(
        pg_pool: &PgPool,
        dataset_id: Uuid,
        validation_samples: i64,
        max_seconds: f32,
        max_text_tokens: i32,
        plbert_languages: &Vec<String>,
        seed: u64,
    ) -> anyhow::Result<Self> {
        let id = Uuid::new_v4();
        println!("initializing session {id}");

        println!(
            "fetching {} validation rows from dataset {}",
            validation_samples, dataset_id
        );
        let validation_rows = fetch_validation_samples(
            pg_pool,
            validation_samples,
            &dataset_id,
            max_seconds,
            max_text_tokens,
        )
        .await?;

        let validation_ids: Vec<Uuid> = validation_rows.iter().map(|r| r.audio_id).collect();

        let validation_histogram = HistogramSampler::from_samples(
            validation_rows,
            plbert_languages,
            max_seconds as f64,
            seed,
        )?;

        println!("fetching training rows from dataset {dataset_id}");
        let training_rows = fetch_training_samples(
            pg_pool,
            &validation_ids,
            &dataset_id,
            max_seconds,
            max_text_tokens,
        )
        .await?;
        let training_histogram = HistogramSampler::from_samples(
            training_rows,
            plbert_languages,
            max_seconds as f64,
            seed,
        )?;

        Ok(Session {
            validation_sampler: validation_histogram,
            training_sampler: training_histogram,
            id,
        })
    }
}
