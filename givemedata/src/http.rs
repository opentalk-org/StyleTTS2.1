use std::net::{Ipv4Addr, SocketAddr};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::get,
};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    session::DataConfig,
    trainings::{Training, TrainingState, TrainingStore},
};

#[derive(Deserialize)]
struct CreateTrainingRequest {
    data_config: DataConfig,
    train_config: String,
}

#[derive(Serialize)]
struct CreateTrainingResponse {
    run_id: Uuid,
    state: TrainingState,
}

#[derive(Serialize)]
struct TrainingResponse {
    run_id: Uuid,
    data_config: DataConfig,
    train_config: String,
    state: TrainingState,
    version: u64,
}

pub async fn serve(
    port: u16,
    trainings: TrainingStore,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/trainings", get(list_trainings).post(create_training))
        .route("/trainings/{run_id}", get(get_training))
        .with_state(trainings);
    let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, port));
    let listener = tokio::net::TcpListener::bind(address).await?;
    info!(%address, "HTTP server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await?;
    Ok(())
}

async fn create_training(
    State(trainings): State<TrainingStore>,
    Json(request): Json<CreateTrainingRequest>,
) -> Result<(StatusCode, Json<CreateTrainingResponse>), StatusCode> {
    let run_id = trainings
        .create(&request.data_config, &request.train_config)
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "creating training failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    info!(run = %run_id, "training created");
    Ok((
        StatusCode::CREATED,
        Json(CreateTrainingResponse {
            run_id,
            state: TrainingState::AwaitingTraining,
        }),
    ))
}

async fn get_training(
    State(trainings): State<TrainingStore>,
    Path(run_id): Path<Uuid>,
) -> Result<Json<TrainingResponse>, StatusCode> {
    let training = trainings.get(run_id).await.map_err(|err| {
        error!(
            run = %run_id,
            error = format!("{err:#}"),
            "getting training failed"
        );
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    match training {
        Some(training) => Ok(Json(training.into())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn list_trainings(
    State(trainings): State<TrainingStore>,
) -> Result<Json<Vec<TrainingResponse>>, StatusCode> {
    let trainings = trainings.list().await.map_err(|err| {
        error!(error = format!("{err:#}"), "listing trainings failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(trainings.into_iter().map(Into::into).collect()))
}

impl From<Training> for TrainingResponse {
    fn from(training: Training) -> Self {
        Self {
            run_id: training.id,
            data_config: training.data_config,
            train_config: training.train_config,
            state: training.state,
            version: training.version,
        }
    }
}
