use std::net::{Ipv4Addr, SocketAddr};

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    session::DataConfig,
    trainings::{TrainingState, TrainingStore},
};

#[derive(Deserialize)]
struct CreateTrainingRequest {
    data_config: DataConfig,
    train_config: String,
}

#[derive(Serialize)]
struct CreateTrainingResponse {
    training_id: Uuid,
    state: TrainingState,
}

pub async fn serve(
    port: u16,
    trainings: TrainingStore,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/trainings", post(create_training))
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
    let training_id = trainings
        .create(&request.data_config, &request.train_config)
        .await
        .map_err(|err| {
            error!(error = format!("{err:#}"), "creating training failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    info!(training = %training_id, "training created");
    Ok((
        StatusCode::CREATED,
        Json(CreateTrainingResponse {
            training_id,
            state: TrainingState::AwaitingTraining,
        }),
    ))
}
