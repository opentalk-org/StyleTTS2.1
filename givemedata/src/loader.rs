use std::path::{Path, PathBuf};

use bytes::{BufMut, Bytes, BytesMut};
use tokio::fs;
use tracing::{debug, trace};

use crate::{audio, sampling};

pub struct LoadedSample {
    pub wave_path: PathBuf,
    pub duration: f64,
    pub speaker_id: i64,
    pub language_id: i32,
    pub text: Bytes,
}

pub type LoadedBatch = Vec<LoadedSample>;

#[derive(Clone)]
pub struct Loader {
    s3_client: aws_sdk_s3::Client,
    bucket: &'static str,
    cache_dir: &'static Path,
}

impl Loader {
    pub fn new(
        s3_client: aws_sdk_s3::Client,
        bucket: &'static str,
        cache_dir: &'static Path,
    ) -> Self {
        Self {
            s3_client,
            bucket,
            cache_dir,
        }
    }

    pub async fn load_sample_bytes(&self, sample: &sampling::Sample) -> anyhow::Result<Bytes> {
        trace!(
            audio = %sample.audio_id,
            object = %sample.object.path,
            offset = sample.object.offset,
            length = sample.object.length,
            "fetching audio from bucket"
        );
        let obj = self
            .s3_client
            .get_object()
            .bucket(self.bucket)
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
        let wave = audio::process_audio(buff.freeze(), 24_000)?;

        Ok(wave)
    }

    pub async fn load_sample(&self, sample: sampling::Sample) -> anyhow::Result<LoadedSample> {
        let wave = self.load_sample_bytes(&sample).await?;

        let path = self
            .cache_dir
            .join(format!("{}-{}.raw", sample.audio_id, uuid::Uuid::new_v4()));
        fs::write(&path, wave).await?;

        Ok(LoadedSample {
            wave_path: path,
            duration: sample.duration,
            speaker_id: sample.speaker_id as i64,
            language_id: sample.language_id,
            text: sample.text,
        })
    }

    pub async fn load_batch(
        &self,
        batch: Vec<crate::sampling::Sample>,
    ) -> anyhow::Result<LoadedBatch> {
        debug!(samples = batch.len(), "loading batch");

        let handles: Vec<_> = batch
            .into_iter()
            .map(|sample| {
                let loader = self.clone();
                tokio::spawn(async move { loader.load_sample(sample).await })
            })
            .collect();

        let mut loaded_batch: Vec<LoadedSample> = vec![];
        for handle in handles {
            loaded_batch.push(handle.await??);
        }

        Ok(loaded_batch)
    }
}
