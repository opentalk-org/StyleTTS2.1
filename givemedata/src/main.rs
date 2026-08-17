mod symbols;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
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
    with afs as (
    	select
    		af.id as audio_id,
    		af.duration,
    		af.language,
    		af.metadata
    	from audio_files af
    	where af.id = any((WITH eligible AS (
    SELECT a.id, a.duration
    FROM audio_files a
    JOIN dataset_audio_files da ON da.audio_file_id = a.id
    WHERE da.dataset_id = $1
      AND a.virtual IS FALSE
      AND a.duration > 0
      AND EXISTS (
          SELECT 1 FROM segments s
          WHERE s.audio_file_id = a.id
            AND btrim(s.phon) <> ''
            AND s.start_seconds < s.end_seconds
      )
)
SELECT id
FROM eligible
WHERE duration >= (
    SELECT percentile_disc($2) WITHIN GROUP (ORDER BY duration)
    FROM eligible
)
ORDER BY duration ASC, id ASC
LIMIT $3))
    	and not af.virtual
    	and af.duration > 0
    	and af.duration < $4
    	order by af.id desc
    ),
    text as (
    	select
    		afs.*,
    		case when jsonb_array_length(jsonb_agg(s.metadata->'_source'->'annotations'->>'speaker_id')) = 1
    			then json_agg(s.metadata->'_source'->'annotations'->>'speaker_id')->>-1
    			else null
    		end as speaker_id,
    		array_to_string(array_agg(s.phon order by s.start_seconds, s.end_seconds, s.id), ' ') as text
    	from afs
    	join segments s on s.audio_file_id = afs.audio_id
    	where trim(s.phon) != ''
    	and s.start_seconds < s.end_seconds
    	group by afs.audio_id, afs.duration, afs.language, afs.metadata
    )
    select
    	audio_id,
    	duration,
    	language,
    	speaker_id,
    	metadata->>'source_id' as source_id,
    	metadata->>'repository' as repository,
    	text
    from text
    where length(text.text) <= $5
    ",
        dataset_id,
        0.9,
        n,
        max_audio_duration as f64,
        max_text_tokens
        ).fetch_all(db).await.context("failed to fetch validation samples")?;

    Ok(rows)
}

struct Sample {
    audio_id: Uuid,
    duration: f64,
    language_id: i32,
    speaker_id: u64,
    text: Vec<i64>,
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
                .ok_or_else(|| Status::internal("training audio is missing its language"))?
                as i32
        };
        let mut hasher = Blake2bVar::new(8).map_err(|_| Status::internal("internal error"))?;
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
        })
    }
}

struct Session {
    validation_samples: Vec<Sample>,
}

type SessionsMap = Arc<RwLock<HashMap<String, Arc<Mutex<Session>>>>>;

struct GiveMeData {
    db_pool: PgPool,
    sessions: SessionsMap,
}

impl GiveMeData {
    async fn new() -> Result<Self, sqlx::Error> {
        let db_pool = PgPool::connect(DATABASE_URL).await?;
        Ok(GiveMeData {
            sessions: Default::default(),
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

        let mut text_cleaner = TextCleaner::default();
        let mut validation_samples = vec![];
        for row in validation_rows {
            match row {
                SampleRow { language: None, .. } => {
                    return Err(Status::internal("training audio is missing its language"));
                }
                SampleRow {
                    audio_id,
                    duration,
                    language: Some(lang),
                    text: Some(text),
                    speaker_id,
                    ..
                } => validation_samples.push(
                    Sample::new(
                        audio_id,
                        duration,
                        text,
                        &mut text_cleaner,
                        &lang,
                        &request.plbert_languages,
                        speaker_id,
                    )
                    .map_err(|e| Status::internal(e.to_string()))?,
                ),

                _ => {}
            }
        }

        let id = Uuid::new_v4().to_string();
        let session = Session { validation_samples };
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
            async move {
                if let Err(err) = data_handler(sessions, &mut stream, &out_tx).await {
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
        let session = session_mux.lock().await;

        if req.split() == Split::Validation {
            // yield validation
            resp_stream.send(Ok(DataResponse {
                data: "validation".to_string(),
            }))?;
        } else {
            // yield traiinig
            resp_stream.send(Ok(DataResponse {
                data: "training".to_string(),
            }))?;
        }
    }

    Ok(())
}

async fn serve() -> anyhow::Result<()> {
    let addr = "0.0.0.0:8080".parse()?;
    println!("[givemedata] listening on 0.0.0.0:8080");
    Server::builder()
        .add_service(GiveMeDataServer::new(GiveMeData::new().await?))
        .serve(addr)
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    serve().await
}
