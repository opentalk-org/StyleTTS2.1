use rand::{SeedableRng, rngs::SmallRng};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{db::fetch_validation_samples, sampling::HistogramSampler};

pub struct Session {
    pub validation_histogram: HistogramSampler,
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

        let rng = SmallRng::from_rng(&mut rand::rng());
        let validation_histogram = HistogramSampler::from_samples(
            validation_rows,
            plbert_languages,
            max_seconds as f64,
            rng,
        )?;

        Ok(Session {
            validation_histogram,
            id,
        })
    }
}
