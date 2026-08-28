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

pub trait Sampler: Send {
    /// `Ok(None)` means the sampler is exhausted and the stream should end.
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>>;
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

/// `sample_rows` should be sorted by duration
pub fn bins_from_rows(
    sample_rows: Vec<SampleRow>,
    plbert_languages: &[String],
) -> anyhow::Result<Vec<DurationBin>> {
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

    Ok(bins)
}

impl HistogramSampler {
    pub fn new(template: Vec<DurationBin>, max_seconds: f64, seed: u64) -> Self {
        // a sample longer than the packing budget can never join a batch;
        // keep it out so it doesn't distort bin weights (mirrors the SQL
        // `duration < max_seconds` filter, applied here per sequence)
        let bins: Vec<DurationBin> = template
            .into_iter()
            .filter_map(|mut bin| {
                bin.samples.retain(|s| s.duration < max_seconds);
                if bin.samples.is_empty() {
                    return None;
                }
                bin.total_seconds = bin.samples.iter().map(|s| s.duration).sum();
                Some(bin)
            })
            .collect();

        Self {
            template: bins.clone(),
            bins,
            max_seconds,
            rng: SmallRng::seed_from_u64(seed),
            seed,
            loops: 0,
        }
    }

    fn sample_batch(&mut self) -> anyhow::Result<Vec<Sample>> {
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

/// Loops its sample set forever; never exhausts.
impl Sampler for HistogramSampler {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        Ok(Some(self.sample_batch()?))
    }
}

struct Sequence {
    batches: u64,
    max_seconds: f64,
}

/// Serves the whole schedule as one stream: exactly `batches` batches with each
/// sequence's shape, then the next sequence, then `None` when the schedule is done.
pub struct ScheduledSampler {
    template: Vec<DurationBin>,
    schedule: std::vec::IntoIter<Sequence>,
    seed: u64,
    current: Option<(HistogramSampler, u64)>,
}

impl ScheduledSampler {
    pub fn new(
        template: Vec<DurationBin>,
        sequences: &[session::SequenceConfig],
        seed: u64,
    ) -> Self {
        let schedule: Vec<Sequence> = sequences
            .iter()
            .map(|s| Sequence {
                batches: s.batches,
                max_seconds: s.max_seconds as f64,
            })
            .collect();
        Self {
            template,
            schedule: schedule.into_iter(),
            seed,
            current: None,
        }
    }
}

impl Sampler for ScheduledSampler {
    fn next_batch(&mut self) -> anyhow::Result<Option<Vec<Sample>>> {
        loop {
            let needs_advance = matches!(&self.current, None | Some((_, 0)));
            if needs_advance {
                self.current = match self.schedule.next() {
                    Some(sequence) => {
                        tracing::info!(
                            batches = sequence.batches,
                            max_seconds = sequence.max_seconds,
                            "starting batch sequence"
                        );
                        Some((
                            HistogramSampler::new(
                                self.template.clone(),
                                sequence.max_seconds,
                                self.seed,
                            ),
                            sequence.batches,
                        ))
                    }
                    None => return Ok(None),
                };
                continue;
            }

            let (sampler, remaining) = self.current.as_mut().expect("current sequence set");
            let batch = sampler.sample_batch()?;
            *remaining -= 1;
            return Ok(Some(batch));
        }
    }
}
