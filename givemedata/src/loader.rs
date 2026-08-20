use bytes::{BufMut, BytesMut};

use crate::{audio, sampling, server::givemedata};

// just an alias for now, will probably change in the future
type LoadedSample = givemedata::Sample;

#[derive(Clone)]
pub struct Loader {
    s3_client: aws_sdk_s3::Client,
    bucket: String,
}

impl Loader {
    pub fn new(s3_client: aws_sdk_s3::Client, bucket: String) -> Self {
        Self { s3_client, bucket }
    }

    pub async fn load_sample(&self, sample: sampling::Sample) -> anyhow::Result<LoadedSample> {
        println!(
            "getting audio file {} from {}:{}-{}",
            sample.audio_id,
            sample.object.path,
            sample.object.offset,
            sample.object.offset + sample.object.length - 1
        );
        let obj = self
            .s3_client
            .get_object()
            .bucket(&self.bucket)
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

        Ok(givemedata::Sample {
            wave,
            duration: sample.duration,
            speaker_id: sample.speaker_id as i64,
            language_id: sample.language_id,
            text: sample.text,
        })
    }

    pub async fn load_batch(
        &self,
        batch: Vec<crate::sampling::Sample>,
    ) -> anyhow::Result<Vec<givemedata::Sample>> {
        println!("loading batch of {}", batch.len());
        let mut loaded_batch: Vec<givemedata::Sample> = vec![];

        let mut set = tokio::task::JoinSet::new();
        for sample in batch {
            let loader = self.clone();
            set.spawn(async move { loader.load_sample(sample).await });
        }
        for res in set.join_all().await {
            loaded_batch.push(res?);
        }

        Ok(loaded_batch)
    }
}
