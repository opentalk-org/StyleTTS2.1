mod symbols;

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Credentials;
use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use bytes::{BufMut, BytesMut};
use hound::WavReader;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::{IndexedMutRandom, IndexedRandom};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use sqlx::PgPool;
use symbols::{TextCleaner, boundary_token_id, text_to_tensor};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::{Request, Response, Status, Streaming, transport::Server};
use uuid::Uuid;

pub mod givemedata {
    tonic::include_proto!("_");
}

use givemedata::give_me_data_server::{GiveMeData as GiveMeDataService, GiveMeDataServer};
use givemedata::{DataRequest, DataResponse, InitRequest, InitResponse, Split};

const DATABASE_URL: &str = "postgres://runflow:runflow@localhost:5433/runflow";

struct SampleRow {
    audio_id: Uuid,
    duration: f64,
    language: Option<String>,
    speaker_id: Option<String>,
    source_id: Option<String>,
    repository: Option<String>,
    text: Option<String>,

    lower_bound: Option<f64>,
    upper_bound: Option<f64>,

    object_path: String,
    byte_offset: i64,
    byte_length: i64,
}

async fn fetch_validation_samples(
    db: &PgPool,
    n: i64,
    dataset_id: &Uuid,
    max_audio_duration: f32,
    max_text_tokens: i32,
) -> anyhow::Result<Vec<SampleRow>> {
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
    .await
    .context("failed to fetch validation samples")?;

    Ok(rows)
}

struct SampleObject {
    path: String,
    offset: i64,
    length: i64,
}

struct Sample {
    audio_id: Uuid,
    duration: f64,
    language_id: i32,
    speaker_id: u64,
    text: Vec<i64>,

    object: SampleObject,
}

struct DurationBin {
    lower_bound: f64,
    upper_bound: f64,
    samples: Vec<Sample>,
    total_seconds: f64,
}

impl Sample {
    fn new(
        audio_id: Uuid,
        duration: f64,
        text: String,
        text_cleaner: &mut TextCleaner,
        language: &String,
        plbert_langs: &Vec<String>,
        speaker_id: Option<String>,
        object: SampleObject,
    ) -> anyhow::Result<Self> {
        let boundary_token_id = boundary_token_id(text_cleaner)?;
        let text_tensor = text_to_tensor(text_cleaner, boundary_token_id, &text);
        let lang_norm = language.trim().to_lowercase().replace("_", "-");
        let language_id: i32 = if plbert_langs.is_empty() {
            0
        } else {
            plbert_langs
                .iter()
                .position(|l| {
                    l == &lang_norm
                        || l == lang_norm
                            .split_once("-")
                            .unwrap_or((lang_norm.as_str(), ""))
                            .0
                })
                .ok_or_else(|| anyhow!("training audio is missing its language"))?
                as i32
        };
        let mut hasher = Blake2bVar::new(8)?;
        let speaker = speaker_id.unwrap_or("0".to_string());
        // TODO: check the ylacombe/expresso thing too
        hasher.update(speaker.as_bytes());
        let mut digest = [0u8; 8];
        hasher
            .finalize_variable(&mut digest)
            .map_err(|e| Status::internal(e.to_string()))?;

        let speaker_id = u64::from_be_bytes(digest) % ((1u64 << 63) - 1);

        Ok(Sample {
            audio_id,
            duration,
            language_id,
            speaker_id,
            text: text_tensor,
            object,
        })
    }
}

fn make_bins(
    sample_rows: Vec<SampleRow>,
    plbert_languages: &Vec<String>,
) -> anyhow::Result<Vec<DurationBin>> {
    let mut bins: Vec<DurationBin> = vec![];

    let mut text_cleaner = TextCleaner::default();

    // sample_rows should be sorted by duration at this point
    for row in sample_rows {
        match row {
            SampleRow {
                audio_id,
                duration,
                language: Some(lang),
                text: Some(text),
                speaker_id,
                lower_bound: Some(lower_bound),
                upper_bound: Some(upper_bound),
                object_path,
                byte_offset,
                byte_length,
                ..
            } => {
                let sample = Sample::new(
                    audio_id,
                    duration,
                    text,
                    &mut text_cleaner,
                    &lang,
                    plbert_languages,
                    speaker_id,
                    SampleObject {
                        path: object_path,
                        offset: byte_offset,
                        length: byte_length,
                    },
                )?;

                if let Some(bin) = bins
                    .iter_mut()
                    .find(|b| b.lower_bound == lower_bound && b.upper_bound == upper_bound)
                {
                    bin.samples.push(sample);
                    bin.total_seconds += duration;
                } else {
                    let bin = DurationBin {
                        lower_bound,
                        upper_bound,
                        total_seconds: duration,
                        samples: vec![sample],
                    };
                    bins.push(bin);
                }
            }
            SampleRow { language: None, .. } => {
                bail!("training audio is missing its language");
            }
            _ => {}
        }
    }

    Ok(bins)
}

struct Session {
    validation_bins: Vec<DurationBin>,
    max_seconds: f64,
}

type SessionsMap = Arc<RwLock<HashMap<String, Arc<Mutex<Session>>>>>;

struct GiveMeData {
    db_pool: PgPool,
    s3_client: aws_sdk_s3::Client,
    sessions: SessionsMap,
}

impl GiveMeData {
    async fn new(s3_client: aws_sdk_s3::Client) -> Result<Self, sqlx::Error> {
        let db_pool = PgPool::connect(DATABASE_URL).await?;
        Ok(GiveMeData {
            sessions: Default::default(),
            s3_client,
            db_pool,
        })
    }
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

        let validation_rows = fetch_validation_samples(
            &self.db_pool,
            request.validation_samples as i64,
            &dataset_id,
            request.max_seconds,
            request.max_text_tokens,
        )
        .await
        .map_err(|e| Status::internal(e.to_string()))?;

        let validation_bins = make_bins(validation_rows, &request.plbert_languages)
            .map_err(|e| Status::internal(e.to_string()))?;

        let id = Uuid::new_v4().to_string();
        let session = Session {
            validation_bins,
            max_seconds: request.max_seconds as f64,
        };
        self.sessions
            .write()
            .await
            .insert(id.clone(), Arc::new(Mutex::new(session)));

        Ok(Response::new(InitResponse { session_id: id }))
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
            let s3_client = self.s3_client.clone();
            let mut rng = SmallRng::from_rng(&mut rand::rng());
            async move {
                if let Err(err) =
                    data_handler(sessions, &mut stream, &out_tx, &s3_client, &mut rng).await
                {
                    println!("erra: {err}");
                }
            }
        });

        Ok(UnboundedReceiverStream::new(out_rx).into())
    }
}

async fn data_handler(
    sessions: SessionsMap,
    req_stream: &mut Streaming<DataRequest>,
    resp_stream: &UnboundedSender<Result<DataResponse, Status>>,
    s3_client: &aws_sdk_s3::Client,
    rng: &mut SmallRng,
) -> anyhow::Result<()> {
    while let Some(req) = req_stream.message().await? {
        let session_mux = {
            let sessions = sessions.read().await;

            if let Some(sess) = sessions.get(&req.session_id).cloned() {
                sess
            } else {
                resp_stream.send(Err(Status::not_found("unknown session")))?;
                return Ok(());
            }
        };
        let mut session = session_mux.lock().await;

        if req.split() == Split::Validation {
            let mut remaining = session.max_seconds;
            let mut batch: Vec<Sample> = vec![];
            let bin = session
                .validation_bins
                .choose_weighted_mut(rng, |b| b.total_seconds)?;

            // TODO: parallel and shit
            loop {
                let eligable: Vec<(usize, f64)> = bin
                    .samples
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        if s.duration <= remaining {
                            Some((i, s.duration))
                        } else {
                            None
                        }
                    })
                    .collect();
                if eligable.is_empty() {
                    break;
                }
                let random_sample = eligable.choose_weighted(rng, |s| s.1)?;
                batch.push(bin.samples.remove(random_sample.0));
                remaining -= random_sample.1;
            }

            let mut loaded_batch: Vec<givemedata::Sample> = vec![];
            for sample in batch {
                let obj = s3_client
                    .get_object()
                    .bucket("runflow")
                    .key(&sample.object.path)
                    .range(format!(
                        "bytes={}-{}",
                        sample.object.offset,
                        sample.object.offset + sample.object.length - 1
                    ))
                    .send()
                    .await?;

                let mut stream = obj.body;

                let mut buff = BytesMut::new();
                while let Some(bytes) = stream.try_next().await? {
                    buff.put(bytes);
                }
                let data = buff.freeze();

                let mut reader = WavReader::new(Cursor::new(data))?;
                let spec = reader.spec();

                let samples = match spec.sample_format {
                    hound::SampleFormat::Float => reader
                        .samples::<f32>()
                        .step_by(spec.channels as usize)
                        .collect::<Result<Vec<_>, _>>()?,

                    hound::SampleFormat::Int => {
                        let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
                        reader
                            .samples::<i32>()
                            .step_by(spec.channels as usize)
                            .map(|s| s.map(|v| v as f32 * scale))
                            .collect::<Result<Vec<_>, _>>()?
                    }
                };

                let data = if spec.sample_rate == 24000 {
                    samples
                } else {
                    let mut rs = Fft::<f32>::new(
                        spec.sample_rate as usize,
                        24_000,
                        1024,
                        1,
                        FixedSync::Both,
                    )?;
                    let adapter = InterleavedSlice::new(&samples, 1, samples.len())?;
                    let out = rs.process_all(&adapter, samples.len(), None)?;
                    out.take_data()
                };

                loaded_batch.push(givemedata::Sample {
                    wave: bytemuck::allocation::cast_vec::<f32, u8>(data),
                    duration: sample.duration,
                    speaker_id: sample.speaker_id as i64,
                    language_id: sample.language_id,
                    text: bytemuck::allocation::cast_vec::<i64, u8>(sample.text),
                });
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

async fn serve() -> anyhow::Result<()> {
    let addr = "0.0.0.0:8080".parse()?;
    let s3_addr = "http://127.0.0.1:9000";
    println!("[givemedata] listening on 0.0.0.0:8080");

    let s3_config = aws_config::defaults(BehaviorVersion::latest())
        .endpoint_url(s3_addr)
        .credentials_provider(Credentials::new("", "", None, None, "r2"))
        .load()
        .await;

    let s3_client = aws_sdk_s3::Client::new(&s3_config);

    Server::builder()
        .add_service(GiveMeDataServer::new(GiveMeData::new(s3_client).await?))
        .serve(addr)
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    serve().await
}
