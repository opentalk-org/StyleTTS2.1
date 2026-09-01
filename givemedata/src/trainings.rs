use serde::Serialize;
use sqlx::PgPool;
use uuid::Uuid;

use crate::session::DataConfig;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingState {
    AwaitingTraining,
    Running,
    Finished,
}

pub struct ClaimedTraining {
    pub data_config: DataConfig,
    pub train_config: String,
}

pub enum ClaimResult {
    Claimed(ClaimedTraining),
    NotFound,
    Unavailable(TrainingState),
}

#[derive(sqlx::FromRow)]
struct TrainingIdRow {
    id: Uuid,
}

#[derive(sqlx::FromRow)]
struct ClaimedTrainingRow {
    data_config: String,
    train_config: String,
}

#[derive(sqlx::FromRow)]
struct TrainingStateRow {
    state: String,
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
    pool: PgPool,
}

impl TrainingStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create(
        &self,
        data_config: &DataConfig,
        train_config: &str,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        let data_config = serde_json::to_string(data_config)?;
        let created = sqlx::query_as::<_, TrainingIdRow>(
            "insert into trainings (id, state, data_config, train_config) \
             values ($1, $2, $3::jsonb, $4) \
             returning id",
        )
        .bind(id)
        .bind(TrainingState::AwaitingTraining.as_str())
        .bind(data_config)
        .bind(train_config)
        .fetch_one(&self.pool)
        .await?;
        Ok(created.id)
    }

    pub async fn claim(&self, id: Uuid) -> anyhow::Result<ClaimResult> {
        let claimed = sqlx::query_as::<_, ClaimedTrainingRow>(
            "update trainings set state = $2 \
             where id = $1 and state = $3 \
             returning data_config::text as data_config, train_config",
        )
        .bind(id)
        .bind(TrainingState::Running.as_str())
        .bind(TrainingState::AwaitingTraining.as_str())
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = claimed {
            return Ok(ClaimResult::Claimed(ClaimedTraining {
                data_config: serde_json::from_str(&row.data_config)?,
                train_config: row.train_config,
            }));
        }

        let state =
            sqlx::query_as::<_, TrainingStateRow>("select state from trainings where id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await?;
        match state {
            Some(row) => Ok(ClaimResult::Unavailable(TrainingState::parse(&row.state)?)),
            None => Ok(ClaimResult::NotFound),
        }
    }

    pub async fn reset(&self, id: Uuid) -> anyhow::Result<()> {
        sqlx::query_as::<_, TrainingIdRow>(
            "update trainings set state = $2 where id = $1 and state = $3 returning id",
        )
        .bind(id)
        .bind(TrainingState::AwaitingTraining.as_str())
        .bind(TrainingState::Running.as_str())
        .fetch_one(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn finish(&self, id: Uuid) -> anyhow::Result<()> {
        let finished = sqlx::query_as::<_, TrainingIdRow>(
            "update trainings set state = $2 where id = $1 and state = $3 returning id",
        )
        .bind(id)
        .bind(TrainingState::Finished.as_str())
        .bind(TrainingState::Running.as_str())
        .fetch_optional(&self.pool)
        .await?;
        if finished.is_none() {
            anyhow::bail!("training {id} is not running");
        }
        Ok(())
    }
}
