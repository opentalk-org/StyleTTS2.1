use std::sync::Arc;

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::session::DataConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingState {
    AwaitingTraining,
    Running,
    Finished,
}

pub struct Training {
    pub id: Uuid,
    pub data_config: DataConfig,
    pub train_config: String,
    pub state: TrainingState,
    pub version: u64,
}

pub enum ClaimResult {
    Claimed(DataConfig, String),
    NotFound,
    Unavailable(TrainingState),
}

#[derive(clickhouse::Row, Deserialize, Serialize)]
struct TrainingsRow {
    #[serde(with = "clickhouse::serde::uuid")]
    id: Uuid,
    data_config: String,
    train_config: String,
    state: String,
    version: u64,
}

impl TrainingState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AwaitingTraining => "awaiting_training",
            Self::Running => "running",
            Self::Finished => "finished",
        }
    }

    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "awaiting_training" => Ok(Self::AwaitingTraining),
            "running" => Ok(Self::Running),
            "finished" => Ok(Self::Finished),
            _ => anyhow::bail!("unknown training state {value:?}"),
        }
    }
}

#[derive(Clone)]
pub struct TrainingStore {
    client: clickhouse::Client,
    transitions: Arc<Mutex<()>>,
}

impl TrainingStore {
    pub fn new(client: clickhouse::Client) -> Self {
        Self {
            client,
            transitions: Arc::new(Mutex::new(())),
        }
    }

    pub async fn create(
        &self,
        data_config: &DataConfig,
        train_config: &str,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let row = TrainingsRow {
            id,
            data_config: serde_json::to_string(data_config)?,
            train_config: train_config.to_string(),
            state: TrainingState::AwaitingTraining.as_str().to_string(),
            version: 1,
        };
        self.insert(&row).await?;

        Ok(id)
    }

    pub async fn get(&self, id: Uuid) -> anyhow::Result<Option<Training>> {
        self.current(id).await?.map(Training::try_from).transpose()
    }

    pub async fn list(&self) -> anyhow::Result<Vec<Training>> {
        self.client
            .query("select ?fields from trainings final order by id")
            .fetch_all::<TrainingsRow>()
            .await?
            .into_iter()
            .map(Training::try_from)
            .collect()
    }

    pub async fn claim(&self, id: Uuid) -> anyhow::Result<ClaimResult> {
        let _transition = self.transitions.lock().await;
        let Some(mut row) = self.current(id).await? else {
            return Ok(ClaimResult::NotFound);
        };
        let state = TrainingState::parse(&row.state)?;
        if state != TrainingState::AwaitingTraining {
            return Ok(ClaimResult::Unavailable(state));
        }

        let data_config = serde_json::from_str(&row.data_config)?;
        let train_config = row.train_config.clone();
        row.replace_state(TrainingState::Running)?;
        self.insert(&row).await?;

        Ok(ClaimResult::Claimed(data_config, train_config))
    }

    pub async fn reset(&self, id: Uuid) -> anyhow::Result<()> {
        self.transition(id, TrainingState::Running, TrainingState::AwaitingTraining)
            .await
    }

    pub async fn finish(&self, id: Uuid) -> anyhow::Result<()> {
        self.transition(id, TrainingState::Running, TrainingState::Finished)
            .await
    }

    async fn transition(
        &self,
        id: Uuid,
        expected: TrainingState,
        next: TrainingState,
    ) -> anyhow::Result<()> {
        let _transition = self.transitions.lock().await;
        let mut row = self
            .current(id)
            .await?
            .with_context(|| format!("training {id} does not exist"))?;
        let current = TrainingState::parse(&row.state)?;
        if current != expected {
            anyhow::bail!(
                "training {id} is {}, expected {}",
                current.as_str(),
                expected.as_str()
            );
        }

        row.replace_state(next)?;
        self.insert(&row).await
    }

    async fn current(&self, id: Uuid) -> anyhow::Result<Option<TrainingsRow>> {
        self.client
            .query("select ?fields from trainings final where id = ?")
            .bind(id.to_string())
            .fetch_optional::<TrainingsRow>()
            .await
            .map_err(Into::into)
    }

    async fn insert(&self, row: &TrainingsRow) -> anyhow::Result<()> {
        let mut insert = self.client.insert::<TrainingsRow>("trainings").await?;
        insert.write(row).await?;
        insert.end().await?;
        Ok(())
    }
}

impl TrainingsRow {
    fn replace_state(&mut self, state: TrainingState) -> anyhow::Result<()> {
        self.state = state.as_str().to_string();
        self.version = self
            .version
            .checked_add(1)
            .context("training version overflowed")?;
        Ok(())
    }
}

impl TryFrom<TrainingsRow> for Training {
    type Error = anyhow::Error;

    fn try_from(row: TrainingsRow) -> Result<Self, Self::Error> {
        Ok(Self {
            id: row.id,
            data_config: serde_json::from_str(&row.data_config)?,
            train_config: row.train_config,
            state: TrainingState::parse(&row.state)?,
            version: row.version,
        })
    }
}
