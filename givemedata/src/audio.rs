use std::io::Cursor;

use bytes::{BufMut, Bytes, BytesMut};
use hound::WavReader;
use rubato::{
    Fft, FixedSync, Resampler,
    audioadapter_buffers::{SizeError, direct::InterleavedSlice},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("wav encoding/decoding error: {0}")]
    Wav(#[from] hound::Error),
    #[error("size error: {0}")]
    SizeError(#[from] SizeError),
    #[error("resampler error: {0}")]
    ResamplerError(#[from] rubato::ResamplerConstructionError),
    #[error("resampling error: {0}")]
    ResamplingError(#[from] rubato::ResampleError),
}

pub fn process_audio(raw_wav: Bytes, target_sample_rate: u32) -> Result<Bytes, AudioError> {
    let mut reader = WavReader::new(Cursor::new(raw_wav))?;
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

    let data = if spec.sample_rate == target_sample_rate {
        samples
    } else {
        let mut rs = Fft::<f32>::new(
            spec.sample_rate as usize,
            target_sample_rate as usize,
            1024,
            1,
            FixedSync::Both,
        )?;
        let adapter = InterleavedSlice::new(&samples, 1, samples.len())?;
        let out = rs.process_all(&adapter, samples.len(), None)?;
        out.take_data()
    };

    let mut wave = BytesMut::with_capacity(4 * data.len());
    for s in data {
        wave.put_f32_le(s);
    }

    Ok(wave.freeze())
}
