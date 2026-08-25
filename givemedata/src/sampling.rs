use anyhow::{anyhow, bail};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use bytes::Bytes;
use rand::{
    SeedableRng,
    rngs::SmallRng,
    seq::{IndexedMutRandom, IndexedRandom},
};
use uuid::Uuid;

use crate::{
    db::SampleRow,
    session,
    symbols::{TextCleaner, boundary_token_id, text_to_tensor_bytes},
};

#[derive(Clone)]
pub struct SampleObject {
    pub path: String,
    pub offset: i64,
    pub length: i64,
}

#[derive(Clone)]
pub struct Sample {
    pub duration: f64,
    pub audio_id: Uuid,
    pub language_id: i32,
    pub speaker_id: u64,
    pub text: Bytes,

    pub object: SampleObject,
}

impl Sample {
    fn new(
        audio_id: Uuid,
        duration: f64,
        text: String,
        text_cleaner: &mut TextCleaner,
        language: &String,
        plbert_langs: &[String],
        speaker_id: Option<String>,
        object: SampleObject,
    ) -> anyhow::Result<Self> {
        let boundary_token_id = boundary_token_id(text_cleaner)?;
        let text_tensor = text_to_tensor_bytes(text_cleaner, boundary_token_id, &text);
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
        hasher.finalize_variable(&mut digest)?;

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

#[derive(Clone)]
pub struct DurationBin {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub samples: Vec<Sample>,
    pub total_seconds: f64,
}

pub struct HistogramSampler {
    template: Vec<DurationBin>,
    bins: Vec<DurationBin>,
    max_seconds: f64,
    rng: SmallRng,
    seed: u64,
    // like a epoch number
    loops: u64,
}

impl HistogramSampler {
    /// `sample_rows` should be sorted by duration
    pub fn from_samples(
        sample_rows: Vec<SampleRow>,
        config: session::Config,
        plbert_languages: &[String],
    ) -> anyhow::Result<Self> {
        let mut bins: Vec<DurationBin> = vec![];

        let mut text_cleaner = TextCleaner::default();

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

        Ok(Self {
            template: bins.clone(),
            bins,
            max_seconds: config.max_seconds as f64,
            rng: SmallRng::seed_from_u64(config.seed),
            seed: config.seed,
            loops: 0,
        })
    }

    pub fn next_batch(&mut self) -> anyhow::Result<Vec<Sample>> {
        if self.bins.iter().all(|b| b.samples.is_empty()) {
            self.loops += 1;
            tracing::info!(loops = self.loops, "bins drained, looping with a new seed");
            self.bins = self.template.clone();
            self.rng = SmallRng::seed_from_u64(self.seed + self.loops);
        }

        let mut remaining = self.max_seconds;
        let mut batch: Vec<Sample> = vec![];
        let bin = self
            .bins
            .choose_weighted_mut(&mut self.rng, |b| b.total_seconds)?;

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
            let random_sample = eligable.choose_weighted(&mut self.rng, |s| s.1)?;
            batch.push(bin.samples.remove(random_sample.0));
            bin.total_seconds = (bin.total_seconds - random_sample.1).max(0.0);
            remaining -= random_sample.1;
        }

        Ok(batch)
    }
}
