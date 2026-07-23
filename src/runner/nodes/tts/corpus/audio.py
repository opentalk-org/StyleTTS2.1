from uuid import NAMESPACE_URL, uuid5

import numpy as np

from runner.nodes.models import Audio, stable_id
from runner.nodes.tts.audio_out import wav_bytes_from_samples
from runner.nodes.tts.corpus.models import CorpusJob
from shared.audio_annotations import AudioAnnotations


def corpus_audio(
    job: CorpusJob,
    samples: np.ndarray,
    sample_rate: int,
) -> Audio:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    wav_bytes = wav_bytes_from_samples(waveform, sample_rate)
    audio_id = stable_id("tts_corpus_audio", job.source_key)
    metadata = {
        "tts_source_key": job.source_key,
        "tts_dataset": job.dataset_name,
        "engine": job.engine.value,
        "voice": job.voice_id,
        "speaker_id": job.speaker_id,
        "language": job.language,
        "text": job.text,
        "stream": job.stream_id,
        "sentence_index": job.sentence_index,
        "sample_rate": sample_rate,
        "byte_length": len(wav_bytes),
    }
    return Audio(
        audio_file_id=uuid5(NAMESPACE_URL, job.source_key),
        name=f"{job.stream_id}-{job.sentence_index:04d}.wav",
        data=wav_bytes,
        sample_rate=sample_rate,
        channels=1,
        start=0.0,
        end=len(waveform) / float(sample_rate),
        annotations=AudioAnnotations(
            speaker_id=job.stream_id,
            metadata=metadata,
        ),
        id=audio_id,
        lineage_id=audio_id,
        byte_length=len(wav_bytes),
    )
