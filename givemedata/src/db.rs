use sqlx::PgPool;
use uuid::Uuid;

pub struct SampleRow {
    pub audio_id: Uuid,
    pub duration: f64,
    pub language: Option<String>,
    pub speaker_id: Option<String>,
    pub source_id: Option<String>,
    pub repository: Option<String>,
    pub text: Option<String>,

    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,

    pub object_path: String,
    pub byte_offset: i64,
    pub byte_length: i64,
}

pub async fn fetch_validation_samples(
    db: &PgPool,
    n: i64,
    dataset_id: &Uuid,
    max_audio_duration: f32,
    max_text_tokens: i32,
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
       speaker_id, source_id, repository, text
from agg
join bucket_files b on b.id = agg.bucket_file_id
where length(text) <= $4
order by duration, audio_id
    ",
        dataset_id,
        n,
        max_audio_duration as f64,
        max_text_tokens
    )
    .fetch_all(db)
    .await?;

    Ok(rows)
}

pub async fn fetch_training_samples(
    db: &PgPool,
    excluded_ids: &[Uuid],
    dataset_id: &Uuid,
    max_audio_duration: f32,
    max_text_tokens: i32,
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
       speaker_id, source_id, repository, text
from agg
join bucket_files b on b.id = agg.bucket_file_id
where length(text) <= $4
order by duration, audio_id
    ",
        dataset_id,
        excluded_ids as &[Uuid],
        max_audio_duration as f64,
        max_text_tokens
    )
    .fetch_all(db)
    .await?;

    Ok(rows)
}
