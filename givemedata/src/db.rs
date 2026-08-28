use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sqlx::PgPool;
use uuid::Uuid;

use crate::session;

pub const VALIDATION_SEED_SALT: u64 = 0x76616c;
pub const TRAINING_SEED_SALT: u64 = 0x747261;

pub struct SampleRow {
    pub audio_id: Uuid,
    pub duration: f64,
    pub language: Option<String>,
    pub speaker_id: Option<String>,
    pub text: Option<String>,

    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,

    pub object_path: String,
    pub byte_offset: i64,
    pub byte_length: i64,
}

pub async fn fetch_validation_samples(
    db: &PgPool,
    config: &session::DataConfig,
) -> Result<Vec<SampleRow>, sqlx::Error> {
    let rows = sqlx::query_as!(
        SampleRow,
        "
with base as (
    select a.*,
           a.metadata ->> 'source_id'  as source_id,
           a.metadata ->> 'repository' as repository
    from audio_files a
    join dataset_audio_files da on da.audio_file_id = a.id
    where da.dataset_id = $1
      and not a.virtual
      and a.duration > 0
      and exists (
          select 1
          from segments s
          where s.audio_file_id = a.id
            and btrim(s.phon) <> ''
            and s.start_seconds < s.end_seconds
      )
),
eligible as (
    select *
    from base
    where duration >= (
        select percentile_disc(0.9) within group (order by duration) from base
    )
    limit $2
),
bounds as (
    select array_agg(power(2, x)) as bounds
    from (
        select generate_series(0, log(2, ceil(max(duration))::int)) as x
        from eligible
    ) g
),
afs as (
    select e.id AS audio_id,
           e.duration,
           e.language,
           e.source_id,
           e.repository,
           e.bucket_file_id,
           e.byte_offset,
           e.byte_length,
           coalesce(b.bounds[w.bin_index], 0) as lower_bound,
           least(power(2, w.bin_index),
                 (select max(duration) from eligible)) as upper_bound
    from eligible e
    cross join bounds b
    cross join lateral (select width_bucket(e.duration, b.bounds)) as w(bin_index)
    where e.duration < $3
),
agg as (
    select a.audio_id,
    	   a.bucket_file_id,
    	   a.byte_offset,
    	   a.byte_length,
           a.duration,
           a.lower_bound,
           a.upper_bound,
           a.language,
           a.source_id,
           a.repository,
           case when count(*) = 1
                then min(s.metadata -> '_source' -> 'annotations' ->> 'speaker_id')
           end as speaker_id,
           array_to_string(
               array_agg(s.phon order by s.start_seconds, s.end_seconds, s.id), ' '
           ) as text
    from afs a
    join segments s ON s.audio_file_id = a.audio_id
    where btrim(s.phon) <> ''
      and s.start_seconds < s.end_seconds
   	group by a.audio_id, a.duration, a.lower_bound, a.upper_bound,
             a.language, a.source_id, a.repository, a.bucket_file_id, a.byte_offset, a.byte_length
)
select audio_id, duration, byte_offset, byte_length, b.path as object_path, lower_bound::float, upper_bound::float, language,
       speaker_id, text
from agg
join bucket_files b on b.id = agg.bucket_file_id
where length(text) <= $4
order by duration, audio_id
    ",
        config.dataset_id,
        config.validation.samples,
        config.validation.max_seconds as f64,
        config.max_text_tokens
    )
    .fetch_all(db)
    .await?;

    Ok(rows)
}

pub async fn fetch_training_samples(
    db: &PgPool,
    excluded_ids: &[Uuid],
    config: &session::DataConfig,
) -> Result<Vec<SampleRow>, sqlx::Error> {
    let rows = sqlx::query_as!(
        SampleRow,
        "
with base as (
    select a.*,
           a.metadata ->> 'source_id'  as source_id,
           a.metadata ->> 'repository' as repository
    from audio_files a
    join dataset_audio_files da on da.audio_file_id = a.id
    where da.dataset_id = $1
      and not a.virtual
      and a.duration > 0
      and exists (
          select 1
          from segments s
          where s.audio_file_id = a.id
            and btrim(s.phon) <> ''
            and s.start_seconds < s.end_seconds
      )
),
eligible as (
    select *
    from base
    where not (id = any($2))
),
bounds as (
    select array_agg(power(2, x)) as bounds
    from (
        select generate_series(0, log(2, ceil(max(duration))::int)) as x
        from eligible
    ) g
),
afs as (
    select e.id AS audio_id,
           e.duration,
           e.language,
           e.source_id,
           e.repository,
           e.bucket_file_id,
           e.byte_offset,
           e.byte_length,
           coalesce(b.bounds[w.bin_index], 0) as lower_bound,
           least(power(2, w.bin_index),
                 (select max(duration) from eligible)) as upper_bound
    from eligible e
    cross join bounds b
    cross join lateral (select width_bucket(e.duration, b.bounds)) as w(bin_index)
    where e.duration < $3
),
agg as (
    select a.audio_id,
    	   a.bucket_file_id,
    	   a.byte_offset,
    	   a.byte_length,
           a.duration,
           a.lower_bound,
           a.upper_bound,
           a.language,
           a.source_id,
           a.repository,
           case when count(*) = 1
                then min(s.metadata -> '_source' -> 'annotations' ->> 'speaker_id')
           end as speaker_id,
           array_to_string(
               array_agg(s.phon order by s.start_seconds, s.end_seconds, s.id), ' '
           ) as text
    from afs a
    join segments s ON s.audio_file_id = a.audio_id
    where btrim(s.phon) <> ''
      and s.start_seconds < s.end_seconds
   	group by a.audio_id, a.duration, a.lower_bound, a.upper_bound,
             a.language, a.source_id, a.repository, a.bucket_file_id, a.byte_offset, a.byte_length
)
select audio_id, duration, byte_offset, byte_length, b.path as object_path, lower_bound::float, upper_bound::float, language,
       speaker_id, text
from agg
join bucket_files b on b.id = agg.bucket_file_id
where length(text) <= $4
order by duration, audio_id
    ",
        config.dataset_id,
        excluded_ids as &[Uuid],
        // fetch once with the widest budget; each sequence narrows in the sampler
        config.training_max_seconds() as f64,
        config.max_text_tokens
    )
    .fetch_all(db)
    .await?;

    Ok(rows)
}

pub fn synthetic_rows(
    max_seconds: f64,
    seed: u64,
    language: &str,
    count: usize,
    seed_salt: u64,
) -> Vec<SampleRow> {
    let mut rng = SmallRng::seed_from_u64(seed ^ seed_salt);

    let mut durations: Vec<f64> = (0..count)
        // log-uniform in [1, max_seconds) so the exp2 bins populate evenly
        .map(|_| rng.random_range(0f64..max_seconds.ln()).exp())
        .collect();
    durations.sort_by(f64::total_cmp);
    let max_duration = durations.last().copied().unwrap_or(1.0);

    durations
        .iter()
        .enumerate()
        .map(|(i, &duration)| {
            let (lower_bound, upper_bound) = exp2_bounds(duration, max_duration);
            SampleRow {
                audio_id: Uuid::from_u128(rng.random()),
                duration,
                language: Some(language.to_string()),
                speaker_id: Some(format!("spk-{}", i % 3)),
                text: Some("wˈʌn tˈuː θrˈiː".to_string()),
                lower_bound: Some(lower_bound),
                upper_bound: Some(upper_bound),
                object_path: "synthetic".to_string(),
                byte_offset: 0,
                byte_length: 0,
            }
        })
        .collect()
}

fn exp2_bounds(duration: f64, max_duration: f64) -> (f64, f64) {
    let top = max_duration.ceil().log2().floor() as i32;
    let bounds: Vec<f64> = (0..=top).map(|x| 2f64.powi(x)).collect();

    let bin_index = bounds
        .iter()
        .take_while(|&&bound| bound <= duration)
        .count();
    let lower = if bin_index == 0 {
        0.0
    } else {
        bounds[bin_index - 1]
    };
    let upper = 2f64.powi(bin_index as i32).min(max_duration);
    (lower, upper)
}
