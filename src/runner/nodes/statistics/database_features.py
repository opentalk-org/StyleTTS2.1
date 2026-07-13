from runflow.core.node import Node
from runflow.policies import BatchMode, BatchPolicy, ResourcePolicy
from runner.nodes.datatypes import AudioPort, JsonPort
from runner.nodes.models import Audio
from runner.nodes.statistics.audio_features import empty_feature_record
from runner.nodes.statistics.segments import speech_segment_records


class DatabaseStatisticsFeaturesNode(Node):
    NODE_TYPE = "DatabaseStatisticsFeatures"
    DESCRIPTION = "Build per-file statistics records from database metadata and stored segments without loading audio bytes, packs, waveforms, or object storage."
    CATEGORY = "Statistics"
    INPUTS = {"audio": AudioPort()}
    OUTPUTS = {"feature_records": JsonPort()}
    BATCH_POLICY = BatchPolicy(BatchMode.MICRO_BATCH, preferred_size=64, max_size=256)
    RESOURCE_POLICY = ResourcePolicy(resources={"io": 1}, keep_loaded=True)

    async def execute(self, batch, context):
        outputs = []
        for inputs in batch:
            context.check_cancel()
            audio = inputs["audio"]
            assert isinstance(audio, Audio), f"unsupported audio input: {type(audio).__name__}"
            assert audio.data is None, f"database statistics received loaded audio bytes: {audio.id}"
            features = empty_feature_record(
                audio,
                audio.sample_rate,
                audio.duration,
                acoustic_metrics_available=False,
            )
            speech = speech_segment_records(audio)
            features["segments"] = speech["segments"]
            features["duplicate_segments_collapsed"] = speech["duplicate_segments_collapsed"]
            outputs.append({"feature_records": features})
        return outputs
